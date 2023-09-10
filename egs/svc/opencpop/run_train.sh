######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

######## Set Experiment Configuration ###########
exp_config="$exp_dir/exp_config.json"
exp_name="opencpop_DDPM_contentvec_test"

num_workers=8
export CUDA_VISIBLE_DEVICES="0" 

######## Train Model ###########
python "${work_dir}"/bins/train.py \
    --config=$exp_config \
    --num_workers=$num_workers \
    --exp_name=$exp_name
