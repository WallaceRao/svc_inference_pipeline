# VC/SVC Recipe: Opencpop

There are five stages in total: data preparation, features extraction, training, inference (conversion), and evaluation.

## 1. Dataset preparation

The official opencpop dataset is released [here](https://wenet.org.cn/opencpop/). The file structure tree is like:

```plaintext
YOUR_PATH/2022-InterSpeech-Opencpop
 ┣ midis
 ┃ ┣ 2001.midi
 ┃ ┣ 2002.midi
 ┃ ┣ 2003.midi
 ┃ ┣ ...
 ┣ segments
 ┃ ┣ wavs
 ┃ ┃ ┣ 2001000001.wav
 ┃ ┃ ┣ 2001000002.wav
 ┃ ┃ ┣ 2001000003.wav
 ┃ ┃ ┣ ...
 ┃ ┣ test.txt
 ┃ ┣ train.txt
 ┃ ┗ transcriptions.txt
 ┣ textgrids
 ┃ ┣ 2001.TextGrid
 ┃ ┣ 2002.TextGrid
 ┃ ┣ 2003.TextGrid
 ┃ ┣ ...
 ┣ wavs
 ┃ ┣ 2001.wav
 ┃ ┣ 2002.wav
 ┃ ┣ 2003.wav
 ┃ ┣ ...
 ┣ TERMS_OF_ACCESS
 ┗ readme.md
```

After downloading, specify the dataset path (eg: `YOUR_PATH/2022-InterSpeech-Opencpop`), the output path for saving the processed data and the training model in  `exp_config.json`:

```json
{
  "base_config": "egs/svc/opencpop/exp_config_base.json",
  "dataset": [
    "opencpop"
  ],
  "dataset_path": {
    // Specify the opencpop dataset path
    "opencpop": "YOUR_PATH/2022-InterSpeech-Opencpop"
  },
  "preprocess": {
    // Specify the output root path to save the processed data 
    "processed_dir": "YOUR_PATH/Amphion/data",
    // Specify the ContentVec pretrained checkpoint file
    "hubert_file": "pretrained/contentvec/checkpoint_best_legacy_500.pt"
  },
  // Specify the output root path to save model ckpts and logs
  "log_dir": "YOUR_PATH/Amphion/logs"
}
```

## 2. Features extraction

To extract acoustic features (eg: mel spectrogram, pitch, and energy) and content features (eg: WeNet, Whisper, and ContentVec):

```bash
sh egs/svc/opencpop/run_preprocess.sh
```

Note that if you want to use WeNet or ContentVec to extract content features, you need to download the pretrained models and specify their paths in `exp_config.json`.

## 3. Training

```bash
sh egs/svc/opencpop/run_train.sh
```

## 4. Inference (Conversion)

## 5. Evaluation
