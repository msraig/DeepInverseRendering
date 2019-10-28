#!/bin/bash

root_dir="./data_root/"
data_dir="${root_dir}/target/"
init_dir="${root_dir}/init/"
log_dir="${root_dir}/results/"

init_dir="${root_dir}/init_refine/"
log_dir_map="${root_dir}/results_refine/"

checkpoint="${root_dir}/model_ae_smooth/"
network="network_ae_fixBN"

cuda_id=0

cd ../optimization/

# rendering loss N20 
CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --N 20 --checkpoint $checkpoint --dataDir $data_dir --logDir "${log_dir}/render_N20/" --initDir  $init_dir --optimize optimize_ae --network $network  --init_method svbrdf --input_type svbrdf --wlv_type random --refine_init image
CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --N 5 --checkpoint $checkpoint --dataDir $data_dir --logDir "${log_dir}/render_N5/" --initDir  $init_dir --optimize $optimize --network $network  --init_method svbrdf --input_type svbrdf --wlv_type random 
CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --N 2 --checkpoint $checkpoint --dataDir $data_dir --logDir "${log_dir}/render_N2/" --initDir  $init_dir --optimize $optimize --network $network   --init_method svbrdf --input_type svbrdf --wlv_type random
CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --N 1 --checkpoint $checkpoint --dataDir $data_dir --logDir "${log_dir}/render_N1/" --initDir  $init_dir --optimize $optimize --network $network  --init_method svbrdf --input_type svbrdf --wlv_type random

python3 mvfile.py --src_path $log_dir --dst_path $init_dir_map --Ns 1,2,5,20

CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --N 20  --checkpoint none  --dataDir $data_dir --logDir "${log_dir_map}/render_N20/" --initial_code  "${init_dir_map}/N20" --optimize $optimize_map --network $network  --scale_size 256 --output_channels 9 --data_loss_type l1log --lr 1e-3 --max_steps 500 --progress_freq 50 --save_freq 50 --data_format npy
CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --N 5  --checkpoint none  --dataDir $data_dir --logDir "${log_dir_map}/render_N5/" --initial_code    "${init_dir_map}/N5" --optimize $optimize_map --network $network  --scale_size 256 --output_channels 9 --data_loss_type l1log --lr 1e-3 --max_steps 500 --progress_freq 50 --save_freq 50 --data_format npy
CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --N 2  --checkpoint none  --dataDir $data_dir --logDir "${log_dir_map}/render_N2/" --initial_code    "${init_dir_map}/N2" --optimize $optimize_map --network $network  --scale_size 256 --output_channels 9 --data_loss_type l1log --lr 1e-3 --max_steps 500 --progress_freq 50 --save_freq 50 --data_format npy
CUDA_VISIBLE_DEVICES=$cuda_id python3 main.py --N 1  --checkpoint none  --dataDir $data_dir --logDir "${log_dir_map}/render_N1/" --initial_code    "${init_dir_map}/N1" --optimize $optimize_map --network $network  --scale_size 256 --output_channels 9 --data_loss_type l1log --lr 1e-3 --max_steps 500 --progress_freq 50 --save_freq 50 --data_format npy

