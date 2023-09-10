# ######## Build Experiment Environment ###########
# exp_dir=$(cd `dirname $0`; pwd)
# work_dir=$(dirname $(dirname $(dirname $exp_dir)))

# export WORK_DIR=$work_dir
# export PYTHONPATH=$work_dir
# export PYTHONIOENCODING=UTF-8


# ######## Set Experiment Configuration ###########
# model_name=diffsvc
# exp_config=$exp_dir/exp_config.json
# log_dir=/mnt/workspace/fangzihao/dev/Amphion/model_ckpt
# exp_name=opencpop_diffsvc_contentvec
# export CUDA_VISIBLE_DEVICES="0" 


# ######## Inference ###########
# python "${work_dir}"/models/svc/"${model_name}"/${model_name}_inference.py \
#     --config "$exp_config" \
#     --checkpoint_dir ${log_dir}/${exp_name}/checkpoints \
#     --vocoder_name "bigvgan" \
#     --checkpoint_dir_of_vocoder '/mnt/data/oss_beijing/pjlab-3090-openmmlabpartner/chenxi.p/SVC/Data-and-Results/model_vocoder/ckpts/bigdata/resume_bigdata_mels_24000hz-audio_bigvgan_bigvgan_pretrained:False_datasetnum:final_finetune_lr_0.0001_dspmatch_True' \
#     --output_dir ${log_dir}/${exp_name}/inference/use_target_mel_min_max \
#     --test_list './opencpopbeta_gloden_test.lst' \
#     --target_singers "opencpop_female1"  \
#     --source_dataset 'opencpop' \
#     --trans_key 'autoshift' \
#     --inference_mode "pndm" \

######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

export CUDA_VISIBLE_DEVICES="0" 

######## Set Experiment Configuration ###########
model_name="diffsvc"
exp_name="m4singer_opencpop_pjs_contentvec"
exp_config="$exp_dir/exp_config.json"
log_dir=/mnt/workspace/chenhaopeng/dev-svc/Amphion/logs

export CUDA_VISIBLE_DEVICES="0"


######## Inference ###########
python "${work_dir}"/bins/inference.py \
    --config "$exp_config" \
    --checkpoint_dir ${log_dir}/${exp_name}/checkpoints \
    --vocoder_name "bigvgan" \
    --checkpoint_dir_of_vocoder '/mnt/data/oss_beijing/pjlab-3090-openmmlabpartner/chenxi.p/SVC/Data-and-Results/model_vocoder/ckpts/bigdata/resume_bigdata_mels_24000hz-audio_bigvgan_bigvgan_pretrained:False_datasetnum:final_finetune_lr_0.0001_dspmatch_True' \
    --output_dir ${log_dir}/${exp_name}/inference/use_target_mel_min_max \
    --test_list './inference_test.lst' \
    --target_singers "opencpop_female1"  \
    --source_dataset 'svcc' \
    --trans_key 'auto' \
    --inference_mode "pndm" \
    --steps 1000