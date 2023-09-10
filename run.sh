work_dir=$(cd `dirname $0`; pwd)

exp_name="opencpop_DDPM_contentvec_conformer"
log_dir="/mnt/workspace/zoulexiao/Amphion/logs"

export CUDA_VISIBLE_DEVICES="0" 
export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir

python bins/inference_file.py \
--acoustics_dir ${log_dir}/${exp_name} \
--vocoder_dir "/mnt/data/oss_beijing/pjlab-3090-openmmlabpartner/chenxi.p/SVC/Data-and-Results/model_vocoder/ckpts/bigdata/resume_bigdata_mels_24000hz-audio_bigvgan_bigvgan_pretrained:False_datasetnum:final_finetune_lr_0.0001_dspmatch_True" \
--target_singers opencpop_female1 \
--trans_key auto
