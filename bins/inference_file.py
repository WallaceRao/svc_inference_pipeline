import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm

from models.svc.diffsvc.diffsvc_inference import DiffSVCInference
from models.svc.transformer.transformer_inference import TransformerInference
from models.vocoders.vocoder_inference import synthesis,load_nnvocoder
from processors.content_extractor import (
    ContentvecExtractor,
    WenetExtractor,
    WhisperExtractor,
)
from utils import f0
from utils.audio import load_audio_torch
from utils.data_utils import *
from utils.data_utils import pitch_shift_to_target
from utils.io import save_audio
from utils.mel import extract_mel_features
from utils.util import load_config

ZERO = 1e-12


def build_parser():
    r"""Build argument parser for inference.py."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="source_audio",
        help="Audio directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="result",
        help="Output directory.",
    )
    parser.add_argument(
        "--acoustics_dir",
        type=str,
        required=False,
        help="Acoustics model checkpoint directory.",
    )
    parser.add_argument(
        "--vocoder_dir",
        type=str,
        required=False,
        help="Vocoder checkpoint directory.",
    )
    parser.add_argument(
        "--target_singers",
        nargs="+",
        help="convert to specific singers",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--trans_key",
        help="0: no pitch shift; autoshift: pitch shift;  int: key shift",
        default=0,
    )
    parser.add_argument(
        "--keep_cache",
        action="store_true",
        default=False,
        help="Keep cache files.",
    )
    return parser


def args2config(work_dir, audio_dir, output_dir):
    r"""Parse inference config into model configs"""
    exp_name="zoulexiao_opencpop_DDPM_contentvec_conformer"
    log_dir="/workspace2/lmxue/data/svc/"
    acoustics_dir = log_dir + "/" + exp_name
    vocoder_dir  = "/workspace2/lmxue/data/vocoder/resume_bigdata_mels_24000hz-audio_bigvgan_bigvgan_pretrained:False_datasetnum:final_finetune_lr_0.0001_dspmatch_True"
    target_singers = "opencpop_female1"
    trans_key = "audo"

    os.putenv("ACOUSTICS_DIR", os.path.abspath(acoustics_dir))
    os.environ["ACOUSTICS_DIR"] = os.path.abspath(acoustics_dir)

    os.putenv("VOCODER_DIR", os.path.abspath(vocoder_dir))
    os.environ["VOCODER_DIR"] = os.path.abspath(vocoder_dir)

    os.putenv("AUDIO_DIR", audio_dir)
    os.environ["AUDIO_DIR"] = audio_dir

    os.putenv("WORK_DIR", work_dir)
    os.environ["WORK_DIR"] = work_dir

    os.putenv("OUTPUT_DIR", output_dir)
    os.environ["OUTPUT_DIR"] = output_dir

    # Load config
    cfg = load_config(os.path.join(acoustics_dir, "checkpoints", "args.json"))
    return cfg


def load_mono_audio(path):
    waveform, sr = load_audio_torch(path, 24000)
    waveform = waveform[None, :]

    assert waveform.dim() == 2 and waveform.size(0) == 1, waveform.size()
    return waveform, sr


def split_audio(
    audio: torch.Tensor, length: int = 8, stride: int = 2, sr=24000
) -> list[torch.Tensor]:
    r"""Split audio into chunks"""

    length = int(length * sr)
    stride = int(stride * sr)
    res = []
    for i in range(0, audio.size(1), stride):
        res.append(audio[..., i : i + length])
        if i + length >= audio.size(1):
            break
    return res


def merge_audio(
    audio_list: list[torch.Tensor], stride: int = 2, sr=24000
) -> torch.Tensor:
    r"""Merge audio chunks"""
    stride = int(stride * sr)

    def _linear_overlap_add(frames: list[torch.Tensor], stride: int):
        # Generic overlap add, with linear fade-in/fade-out, supporting complex scenario
        # e.g., more than 2 frames per position.
        # The core idea is to use a weight function that is a triangle,
        # with a maximum value at the middle of the segment.
        # We use this weighting when summing the frames, and divide by the sum of weights
        # for each positions at the end. Thus:
        #   - if a frame is the only one to cover a position, the weighting is a no-op.
        #   - if 2 frames cover a position:
        #          ...  ...
        #         /   \/   \
        #        /    /\    \
        #            S  T       , i.e. S offset of second frame starts, T end of first frame.
        # Then the weight function for each one is: (t - S), (T - t), with `t` a given offset.
        # After the final normalization, the weight of the second frame at position `t` is
        # (t - S) / (t - S + (T - t)) = (t - S) / (T - S), which is exactly what we want.
        #
        #   - if more than 2 frames overlap at a given point, we hope that by induction
        #      something sensible happens.

        assert len(frames)

        if len(frames) == 1:
            return frames[0]

        device = frames[0].device
        dtype = frames[0].dtype
        shape = frames[0].shape[:-1]
        total_size = stride * (len(frames) - 1) + frames[-1].size(-1)

        sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
        out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
        offset: int = 0

        for frame in frames:
            frame_length = frame.size(-1)
            t = torch.linspace(
                0, 1, frame_length + 2, device=device, dtype=torch.float32
            )[1:-1]
            weight = 0.5 - (t - 0.5).abs()
            weighted_frame = frame * weight

            cur = out[..., offset : offset + frame_length]
            cur += weighted_frame[..., : cur.size(-1)]
            out[..., offset : offset + frame_length] = cur

            cur = sum_weight[offset : offset + frame_length]
            cur += weight[..., : cur.size(-1)]
            sum_weight[offset : offset + frame_length] = cur

            offset += stride
        assert sum_weight.min() > 0
        return out / sum_weight

    audio = _linear_overlap_add(audio_list, stride)
    return audio


def parse_vocoder(vocoder_dir):
    r"""Parse vocoder config"""
    vocoder_dir = os.path.abspath(vocoder_dir)
    ckpt_list = [ckpt for ckpt in Path(vocoder_dir).glob("*.pt")]
    ckpt_list.sort(key=lambda x: int(x.stem), reverse=True)
    ckpt_path = str(ckpt_list[0])
    vocoder_cfg = load_config(os.path.join(vocoder_dir, "args.json"), lowercase=True)
    vocoder_cfg.model.bigvgan = vocoder_cfg.vocoder
    return vocoder_cfg, ckpt_path


def prepare_metadata(audio_dir, temp_dir):
    r"""Prepare metadata for inference"""

    # mkdir for temp(condition) files
    #temp_dir = os.path.join(os.getenv("OUTPUT_DIR"), ".temp")
    os.makedirs(os.path.join(temp_dir, "audio"), exist_ok=True)
    #os.putenv("TEMP_DIR", temp_dir)
    #os.environ["TEMP_DIR"] = temp_dir

    # get source file list
    # audio_dir = os.getenv("AUDIO_DIR")
    # TODO: Support recursive search
    audio_list = [str(audio) for audio in Path(audio_dir).glob("*.wav")]
    audio_list += [str(audio) for audio in Path(audio_dir).glob("*.flac")]
    audio_list += [str(audio) for audio in Path(audio_dir).glob("*.mp3")]

    # prepare metadata to dump as json
    metadata = []

    for audio_path in tqdm(audio_list, desc="Preparing metadata"):
        # load audio, [channels x frames]
        audio, sr = load_mono_audio(audio_path)
        # TODO: Write into YAML, support other than 24kHz
        audio = torchaudio.functional.resample(audio, sr, 24000)

        metadata_chunk = []
        # split audio
        audio_chunks = split_audio(audio)
        for i, audio_chunk in tqdm(enumerate(audio_chunks), leave=False, unit="chunk"):
            source = Path(audio_path).stem
            uid = f"{source}_{i}"

            # save audio chunk
            torchaudio.save(
                os.path.join(temp_dir, "audio", uid + ".wav"),
                audio_chunk,
                24000,
                encoding="PCM_S",
                bits_per_sample=16,
            )

            # save metadata
            metadata_chunk.append(
                {
                    "source": source,
                    "index": i,
                    "uid": uid,
                    "path": os.path.join(temp_dir, "audio", uid + ".wav"),
                    "duration": audio_chunk.size(1) / 24000,
                }
            )
        metadata.append(metadata_chunk)

    # dump metadata
    metadata_path = os.path.join(temp_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False, sort_keys=True)


@torch.no_grad()
def prepare_acoustics_feature(cfg, temp_dir):
    #Atemp_dir = os.getenv("TEMP_DIR")

    metadata_path = os.path.join(temp_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # TODO: Make it parallel, support GPU, with update of preprocess
    for chunks in tqdm(metadata, desc="Extracting acoustics features"):
        if cfg.preprocess.mel_min_max_norm:
            mel_temp = []

        # Collect f0 for file-wise auto pitch shift
        f0_temp = []

        for item in tqdm(chunks, leave=False, unit="chunk"):
            source = item["source"]
            index = item["index"]
            uid = item["uid"]

            audio_path = os.path.join(temp_dir, "audio", uid + ".wav")
            wav_torch, _ = load_mono_audio(audio_path)  # [1 x frames]

            # Extract Mel
            if cfg.preprocess.extract_mel:
                mel = extract_mel_features(wav_torch, cfg.preprocess)
                assert mel.size(0) == cfg.preprocess.n_mel, (
                    f"Mel size mismatched, expected {cfg.preprocess.n_mel}, "
                    f"got {mel.size(0)}"
                )
                # print("Mel size:", mel.size())
                os.makedirs(os.path.join(temp_dir, "mel"), exist_ok=True)
                torch.save(mel, os.path.join(temp_dir, "mel", uid + ".pt"))

                if cfg.preprocess.mel_min_max_norm:
                    mel_temp.append(mel)

            # Extract f0
            if cfg.preprocess.extract_pitch:
                pitch = torch.from_numpy(
                    f0.get_f0(wav_torch.numpy(force=True), cfg.preprocess)
                )
                # print("Pitch size:", pitch.size())
                f0_temp.append(pitch)
                os.makedirs(os.path.join(temp_dir, "pitch"), exist_ok=True)
                torch.save(pitch, os.path.join(temp_dir, "pitch", uid + ".pt"))
                if cfg.preprocess.extract_uv:
                    uv = pitch != 0
                    uv = [0 if frame_uv else 1 for frame_uv in uv]
                    uv = torch.tensor(uv, dtype=torch.int64)
                    os.makedirs(os.path.join(temp_dir, "uvs"), exist_ok=True)
                    torch.save(uv, os.path.join(temp_dir, "uvs", uid + ".pt"))

            # Extract energy
            if cfg.preprocess.extract_energy:
                energy = (mel.exp() ** 2.0).sum(0).sqrt()
                assert energy.size(0) == mel.size(1) and energy.dim() == 1, (
                    f"Energy size mismatched, expected ({wav_torch.size(1)}, ), "
                    f"got {energy.size()}"
                )
                # print("Energy size:", energy.size())
                os.makedirs(os.path.join(temp_dir, "energy"), exist_ok=True)
                torch.save(energy, os.path.join(temp_dir, "energy", uid + ".pt"))

        # Store File-wise mel Min-Max
        if cfg.preprocess.mel_min_max_norm:
            mel_temp = torch.cat(mel_temp, dim=1)
            mel_min = mel_temp.min(dim=1)[0]
            mel_max = mel_temp.max(dim=1)[0]
            assert (
                mel_min.size(0) == mel_max.size(0) == cfg.preprocess.n_mel
                and mel_min.dim() == mel_max.dim() == 1
            ), f"{mel_min.size()}\n{mel_max.size()}"
            os.makedirs(
                os.path.join(temp_dir, "mel_min_max_stats", source), exist_ok=True
            )
            torch.save(
                mel_min,
                os.path.join(temp_dir, "mel_min_max_stats", source, "min.pt"),
            )
            torch.save(
                mel_max,
                os.path.join(temp_dir, "mel_min_max_stats", source, "max.pt"),
            )

        # Store File-wise F0
        file_f0 = torch.cat(f0_temp, dim=0)
        mean = file_f0[file_f0 != 0].mean()
        std = file_f0[file_f0 != 0].std()
        median = file_f0[file_f0 != 0].median()
        os.makedirs(os.path.join(temp_dir, "f0_stats"), exist_ok=True)
        torch.save(
            torch.tensor([mean, std, median]),
            os.path.join(temp_dir, "f0_stats", source + ".pt"),
        )


@torch.no_grad()
def prepare_content_feature(cfg, contentvec_extractor, temp_dir):
    #temp_dir = os.getenv("TEMP_DIR")

    metadata_path = os.path.join(temp_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Whisper
    if cfg.preprocess.extract_whisper_feature:
        print("extract whisper feature")
        whisper_extractor = WhisperExtractor(cfg)
        whisper_extractor.load_model()
        os.makedirs(os.path.join(temp_dir, "whisper"), exist_ok=True)

        utt_batch = []
        for chunks in tqdm(metadata, desc="Extracting whisper features"):
            for item in tqdm(chunks, leave=False, unit="chunk"):
                utt_batch.append({"Path": item["path"], "uid": item["uid"]})
                if len(utt_batch) == cfg.preprocess.whisper_batch_size:
                    whisper_batch = whisper_extractor.extract_content_batch(utt_batch)
                    whisper_batch = whisper_batch.chunk(
                        cfg.preprocess.whisper_batch_size
                    )
                    for i, item in enumerate(whisper_batch):
                        torch.save(
                            item.squeeze(0),
                            os.path.join(
                                temp_dir, "whisper", utt_batch[i]["uid"] + ".pt"
                            ),
                        )
                    utt_batch = []

        if len(utt_batch):
            whisper_batch = whisper_extractor.extract_content_batch(utt_batch)
            whisper_batch = whisper_batch.chunk(len(utt_batch))
            for i, item in enumerate(whisper_batch):
                torch.save(
                    item.squeeze(0),
                    os.path.join(temp_dir, "whisper", utt_batch[i]["uid"] + ".pt"),
                )

    # Contentvec
    if cfg.preprocess.extract_contentvec_feature:
        print("extract contentvec feature")
        os.makedirs(os.path.join(temp_dir, "contentvec"), exist_ok=True)

        utt_batch = []
        for chunks in tqdm(metadata, desc="Extracting contentvec features"):
            for item in tqdm(chunks, leave=False, unit="chunk"):
                utt_batch.append({"Path": item["path"], "uid": item["uid"]})
                if len(utt_batch) == cfg.preprocess.contentvec_batch_size:
                    contentvec_batch = contentvec_extractor.extract_content_batch(
                        utt_batch
                    )
                    contentvec_batch = contentvec_batch.chunk(
                        cfg.preprocess.contentvec_batch_size
                    )
                    for i, item in enumerate(contentvec_batch):
                        torch.save(
                            item.squeeze(0),
                            os.path.join(
                                temp_dir, "contentvec", utt_batch[i]["uid"] + ".pt"
                            ),
                        )
                    utt_batch = []

        if len(utt_batch):
            contentvec_batch = contentvec_extractor.extract_content_batch(utt_batch)
            contentvec_batch = contentvec_batch.chunk(len(utt_batch))
            for i, item in enumerate(contentvec_batch):
                torch.save(
                    item.squeeze(0),
                    os.path.join(temp_dir, "contentvec", utt_batch[i]["uid"] + ".pt"),
                )

    # Wenet
    if cfg.preprocess.extract_wenet_feature:
        print("extract wenet feature")
        wenet_extractor = WenetExtractor(cfg)
        wenet_extractor.load_model()
        os.makedirs(os.path.join(temp_dir, "wenet"), exist_ok=True)

        utt_batch = []
        for chunks in tqdm(metadata, desc="Extracting wenet features"):
            for item in tqdm(chunks, leave=False, unit="chunk"):
                utt_batch.append({"Path": item["path"], "uid": item["uid"]})
                if len(utt_batch) == cfg.preprocess.wenet_batch_size:
                    wenet_batch = wenet_extractor.extract_content_batch(utt_batch)
                    wenet_batch = wenet_batch.chunk(cfg.preprocess.wenet_batch_size)
                    for i, item in enumerate(wenet_batch):
                        torch.save(
                            item.squeeze(0),
                            os.path.join(
                                temp_dir, "wenet", utt_batch[i]["uid"] + ".pt"
                            ),
                        )
                    utt_batch = []

        if len(utt_batch):
            wenet_batch = wenet_extractor.extract_content_batch(utt_batch)
            wenet_batch = wenet_batch.chunk(len(utt_batch))
            for i, item in enumerate(wenet_batch):
                torch.save(
                    item.squeeze(0),
                    os.path.join(temp_dir, "wenet", utt_batch[i]["uid"] + ".pt"),
                )


@torch.no_grad()
def conversion(args, cfg, inference, temp_dir):
    #temp_dir = os.getenv("TEMP_DIR")
    os.makedirs(os.path.join(temp_dir, "pred"), exist_ok=True)

    metadata_path = os.path.join(temp_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    dataset = args.target_singers[0].split("_")[0]
    # TODO: support multiple target singers
    singer = args.target_singers
    data_dir = os.path.join(args.acoustics_dir, "singers.json")
    with open(data_dir, "r") as f:
        singers = json.load(f)
    # for singer in args.target_singers:
    print(
        "-" * 20,
        "\nConversion to {}...\n".format(singer),
    )
    print("use singer:", singer)
    singer_id = singers[singer]
    target_dataset = singer.split("_")[0]
    if args.trans_key:
        statistics_path = os.path.join(
            cfg.preprocess.processed_dir,
            target_dataset,
            cfg.preprocess.pitch_dir,
            "statistics.json",
        )

        target_pitch_median = json.load(open(statistics_path, "r"))[f"{singer}"][
            "voiced_positions"
        ]["median"]

    # TODO: One source audio for one batch, need improvement, with update of inference
    for chunks in tqdm(metadata, desc="Conversion"):
        batch_data = {
            "spk_id": torch.full((len(chunks), 1), singer_id).long().cuda(),
        }
        if cfg.preprocess.extract_mel:
            mels = []
        if cfg.preprocess.extract_pitch:
            pitches = []
            if cfg.preprocess.extract_uv:
                uvs = []
        if cfg.preprocess.extract_energy:
            energies = []
        if cfg.preprocess.extract_whisper_feature:
            whispers = []
            whisper_extractor = WhisperExtractor(cfg)
        if cfg.preprocess.extract_contentvec_feature:
            contentvecs = []
            contentvec_extractor = ContentvecExtractor(cfg.preprocess)
        if cfg.preprocess.extract_wenet_feature:
            wenets = []
            wenet_extractor = WenetExtractor(cfg)

        src_mel_min = torch.load(
            os.path.join(temp_dir, "mel_min_max_stats", chunks[0]["source"], "min.pt")
        )[None, :]
        src_mel_max = torch.load(
            os.path.join(temp_dir, "mel_min_max_stats", chunks[0]["source"], "max.pt")
        )[None, :]
        tgt_mel_min = torch.from_numpy(
            np.load(
                os.path.join(
                    cfg.preprocess.processed_dir,
                    target_dataset,
                    cfg.preprocess.mel_min_max_stats_dir,
                    "mel_min.npy",
                )
            )
        )[None, :]
        tgt_mel_max = torch.from_numpy(
            np.load(
                os.path.join(
                    cfg.preprocess.processed_dir,
                    target_dataset,
                    cfg.preprocess.mel_min_max_stats_dir,
                    "mel_max.npy",
                )
            )
        )[None, :]
        if type(args.trans_key) != int:
            _, _, source_f0_median = torch.load(
                os.path.join(temp_dir, "f0_stats", chunks[0]["source"] + ".pt")
            )

        lengths = []
        for item in tqdm(chunks, desc="Loading...", leave=False, unit="chunk"):
            uid = item["uid"]

            # Load Acoustics Feature
            if cfg.preprocess.extract_mel:
                mel = torch.load(os.path.join(temp_dir, "mel", uid + ".pt")).transpose(
                    0, 1
                )
                assert (
                    mel.size(1)
                    == src_mel_min.size(1)
                    == src_mel_max.size(1)
                    == cfg.preprocess.n_mel
                ), (
                    f"Mel size mismatched, expected {cfg.preprocess.n_mel}, got"
                    f"Mel: {mel.size(1)}"
                    f"src_mel_min: {src_mel_min.size(1)}"
                    f"src_mel_max: {src_mel_max.size(1)}"
                )
                assert mel.dim() == src_mel_min.dim() == src_mel_max.dim() == 2, (
                    f"Mel dim mismatched, expected 2, got"
                    f"Mel: {mel.dim()}"
                    f"src_mel_min: {src_mel_min.dim()}"
                    f"src_mel_max: {src_mel_max.dim()}"
                )
                mel = (mel - src_mel_min) / (
                    src_mel_max - src_mel_min + ZERO
                ) * 2.0 - 1.0
                length = mel.size(0)
                mels.append(mel)
            if cfg.preprocess.extract_pitch:
                pitch = torch.load(os.path.join(temp_dir, "pitch", uid + ".pt"))

                if args.trans_key:
                    # print(self.trans_key, type(self.trans_key))
                    if type(args.trans_key) == int:
                        pitch = transpose_key(pitch, args.trans_key)
                    elif args.trans_key:
                        pitch = pitch_shift_to_target(
                            pitch, target_pitch_median, source_f0_median
                        )

                pitch = torch.from_numpy(align_length(pitch.numpy(force=True), length))
                pitches.append(pitch)
                if cfg.preprocess.extract_uv:
                    uv = torch.load(os.path.join(temp_dir, "uvs", uid + ".pt"))
                    uv = torch.from_numpy(align_length(uv, length))
                    uvs.append(uv)
            if cfg.preprocess.extract_energy:
                energy = torch.load(os.path.join(temp_dir, "energy", uid + ".pt"))
                energy = torch.from_numpy(
                    align_length(energy.numpy(force=True), length)
                )
                energies.append(energy)

            # Load Whisper Feature
            if cfg.preprocess.extract_whisper_feature:
                whisper = torch.load(os.path.join(temp_dir, "whisper", uid + ".pt"))
                whisper = torch.from_numpy(
                    whisper_extractor.offline_align(whisper.numpy(force=True), length)
                )
                whispers.append(whisper)

            # Load Contentvec Feature
            if cfg.preprocess.extract_contentvec_feature:
                contentvec = torch.load(
                    os.path.join(temp_dir, "contentvec", uid + ".pt")
                )
                contentvec = torch.from_numpy(
                    contentvec_extractor.offline_align(
                        contentvec.numpy(force=True), length
                    )
                )
                contentvecs.append(contentvec)

            # Load Wenet Feature
            if cfg.preprocess.extract_wenet_feature:
                wenet = torch.load(os.path.join(temp_dir, "wenet", uid + ".pt"))
                wenet = torch.from_numpy(
                    wenet_extractor.offline_align(wenet.numpy(force=True), length)
                )
                wenets.append(wenet)

            lengths.append(length)

        batch_data["target_len"] = torch.tensor(lengths).long().cuda()
        if cfg.preprocess.extract_mel:
            mels = nn.utils.rnn.pad_sequence(mels, batch_first=True)
            batch_data["mel"] = mels.cuda()
        if cfg.preprocess.extract_pitch:
            pitches = nn.utils.rnn.pad_sequence(pitches, batch_first=True)
            batch_data["frame_pitch"] = pitches.cuda()
            if cfg.preprocess.extract_uv:
                uvs = nn.utils.rnn.pad_sequence(uvs, batch_first=True)
                batch_data["frame_uv"] = uvs.cuda()
        if cfg.preprocess.extract_energy:
            energies = nn.utils.rnn.pad_sequence(energies, batch_first=True)
            batch_data["frame_energy"] = energies.cuda()
        if cfg.preprocess.extract_whisper_feature:
            whispers = nn.utils.rnn.pad_sequence(whispers, batch_first=True)
            batch_data["whisper_feat"] = whispers.cuda()
        if cfg.preprocess.extract_contentvec_feature:
            contentvecs = nn.utils.rnn.pad_sequence(contentvecs, batch_first=True)
            batch_data["contentvec_feat"] = contentvecs.cuda()
        if cfg.preprocess.extract_wenet_feature:
            wenets = nn.utils.rnn.pad_sequence(wenets, batch_first=True)
            batch_data["wenet_feat"] = wenets.cuda()

        y_pred, _ = inference.inference_each_batch(batch_data)
        for i, item in enumerate(y_pred):
            uid = chunks[i]["uid"]
            mel = item.squeeze(0).cpu()
            assert (
                mel.size(1)
                == tgt_mel_min.size(1)
                == tgt_mel_max.size(1)
                == cfg.preprocess.n_mel
            ), (
                f"Mel size mismatched, expected {cfg.preprocess.n_mel}, got"
                f"Mel: {mel.size(1)}"
                f"tgt_mel_min: {tgt_mel_min.size(1)}"
                f"tgt_mel_max: {tgt_mel_max.size(1)}"
            )
            assert mel.dim() == tgt_mel_min.dim() == tgt_mel_max.dim() == 2, (
                f"Mel dim mismatched, expected 1, got"
                f"Mel: {mel.dim()}"
                f"tgt_mel_min: {tgt_mel_min.dim()}"
                f"tgt_mel_max: {tgt_mel_max.dim()}"
            )
            torch.save(
                (mel + 1.0) / 2.0 * (tgt_mel_max - tgt_mel_min + ZERO) + tgt_mel_min,
                os.path.join(temp_dir, "pred", uid + ".pt"),
            )


# TODO: Auto shift pitch
@torch.no_grad()
def acoustics_pred2audio(vocoder_cfg, vocoder_ckpt, vocoder, temp_dir):
    #temp_dir = os.getenv("TEMP_DIR")
    os.makedirs(os.path.join(temp_dir, "audio_pred"), exist_ok=True)
    metadata_path = os.path.join(temp_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    acoustics_pred = []
    for item in tqdm(metadata, desc="Synthesizing Audio"):
        for chunk in tqdm(item, desc="Loading...", leave=False, unit="chunk"):
            uid = chunk["uid"]
            pred = torch.load(os.path.join(temp_dir, "pred", uid + ".pt"))
            acoustics_pred.append(pred)


    audios_pred = synthesis(
        "bigvgan", vocoder, vocoder_cfg, vocoder_ckpt, len(acoustics_pred), acoustics_pred
    )

    i = 0
    for item in tqdm(metadata, desc="Saving Generated Audios"):
        for chunk in tqdm(item, desc="Saving...", leave=False, unit="chunk"):
            uid = chunk["uid"]
            audio_pred = audios_pred[i].unsqueeze(0).cpu()
            i += 1
            torchaudio.save(
                os.path.join(temp_dir, "audio_pred", uid + ".wav"),
                audio_pred,
                24000,
                encoding="PCM_S",
                bits_per_sample=16,
            )

    assert i == len(audios_pred)


def generate_results(temp_dir, output_dir):
    #temp_dir = os.getenv("TEMP_DIR")
    #output_dir = os.getenv("OUTPUT_DIR")
    metadata_path = os.path.join(temp_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    for item in tqdm(metadata, desc="Generating Final Results"):
        result = []
        for chunk in tqdm(item, desc="Generating...", leave=False, unit="chunk"):
            uid = chunk["uid"]
            audio_pred, _ = load_mono_audio(
                os.path.join(temp_dir, "audio_pred", uid + ".wav")
            )
            result.append(audio_pred)
        result = merge_audio(result)
        source = item[0]["source"]
        # torchaudio.save(
        #     os.path.join(output_dir, source + ".wav"),
        #     result,
        #     24000,
        # )
        save_audio(
            os.path.join(output_dir, source + ".wav"),
            result.squeeze(0).numpy(force=True),
            24000,
            add_silence=True,
            turn_up=True,
        )


def do_convert(contentvec_extractor, acoustic_inference, vocoder, vocoder_cfg, vocoder_ckpt, wav_folder, target_singers):
    exp_name="zoulexiao_opencpop_DDPM_contentvec_conformer"
    log_dir="/workspace2/lmxue/data/svc/"
    acoustics_dir = log_dir + "/" + exp_name
    vocoder_dir  = "/workspace2/lmxue/data/vocoder/resume_bigdata_mels_24000hz-audio_bigvgan_bigvgan_pretrained:False_datasetnum:final_finetune_lr_0.0001_dspmatch_True"
    target_singers = target_singers
    trans_key = "audo"
    work_dir = "/workspace2/yonghui/svc_data/work_dir"
    audio_dir = wav_folder + "/audio_dir"
    output_dir = wav_folder + "/output_dir"
    temp_dir = wav_folder + "/temp_dir"

    cfg = args2config(work_dir, audio_dir, output_dir)
    args = build_parser().parse_args()
    args.acoustics_dir = acoustics_dir
    args.target_singers = target_singers
    args.trans_key  = trans_key
    args.audio_dir  = audio_dir
    args.output_dir  = output_dir

    inference = acoustic_inference
    # Prepare metadata, aka. split audio and dump json
    prepare_metadata(audio_dir, temp_dir)
    # Prepare feature
    prepare_acoustics_feature(cfg, temp_dir)
    prepare_content_feature(cfg, contentvec_extractor, temp_dir)
    # ASR conversion
    conversion(args, cfg, inference, temp_dir)
    acoustics_pred2audio(vocoder_cfg, vocoder_ckpt, vocoder, temp_dir)
    # Save to file
    generate_results(temp_dir, output_dir)

    # Clean up
    #if not args.keep_cache:
    #    temp_dir = os.getenv("TEMP_DIR")
    #    print(f"\nRemoving cache files...")
    #    shutil.rmtree(temp_dir)
    #    print("Done!")

def main():
    exp_name="zoulexiao_opencpop_DDPM_contentvec_conformer"
    log_dir="/workspace2/lmxue/data/svc/"
    acoustics_dir = log_dir + "/" + exp_name
    vocoder_dir  = "/workspace2/lmxue/data/vocoder/resume_bigdata_mels_24000hz-audio_bigvgan_bigvgan_pretrained:False_datasetnum:final_finetune_lr_0.0001_dspmatch_True"
    target_singers = "opencpop_female1"
    trans_key = "audo"
    work_dir = "/workspace2/yonghui/svc_data/work_dir"
    audio_dir = "/workspace2/yonghui/svc_data/audio_dir"
    output_dir = "/workspace2/yonghui/svc_data/output_dir"
    cfg = args2config(work_dir, audio_dir, output_dir)
    # preload models
    print("preload models")
    print("preload contentvec extractor")
    contentvec_extractor = ContentvecExtractor(cfg)
    contentvec_extractor.load_model()
    print("preload vocoder model")
    vocoder_cfg, vocoder_ckpt = parse_vocoder(vocoder_dir)
    vocoder_cfg.preprocess = cfg.preprocess
    vocoder_cfg.preprocess.hop_length = vocoder_cfg.preprocess.hop_size
    vocoder = load_nnvocoder(vocoder_cfg, "bigvgan", weights_file=vocoder_ckpt, from_multi_gpu=True)

    print("preload acoustic model")
    inference = None
    args = build_parser().parse_args()
    args.acoustics_dir = acoustics_dir
    args.target_singers = target_singers
    args.trans_key  = trans_key
    args.audio_dir  = audio_dir
    args.output_dir  = output_dir
    args.checkpoint_file = None
    args.checkpoint_dir = os.path.join(os.getenv("ACOUSTICS_DIR"), "checkpoints")
    args.checkpoint_dir_of_vocoder = None
    args.checkpoint_file_of_vocoder = None
    args.inference_mode = "pndm"
    if cfg.model_type == "Transformer":
        inference = TransformerInference(cfg, args)
    elif cfg.model_type == "DiffSVC":
        inference = DiffSVCInference(cfg, args)
    print("preload models finished")
    wav_folder = "/workspace2/yonghui/svc_data/asdf34zx"
    do_convert(contentvec_extractor, inference, vocoder, vocoder_cfg, vocoder_ckpt, wav_folder)

if __name__ == "__main__":
    main()