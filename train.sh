#!/bin/bash
#SBATCH --job-name=deter_tome
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --cpus-per-task=40
#SBATCH --partition=multigpu

fuser -k 29501/tcp

# /nfs/users/ext_cvgroup-9/miniconda3/envs/cv703/bin/python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --isaid_path /nfs/projects/cv703/jazz-cvgroup-9 --output_dir ./Outputs/baseline_continue/  --resume /nfs/users/ext_sanoojan.baliah/Sanoojan/detr/Outputs/baseline/checkpoint0299.pth> Outs/baseline_batch2.out
backbone=resnet101
lr=1e-4
lr_backbone=1e-5
epochs=500
queries=100
Name=baseline

# CUDA_VISIBLE_DEVICES=4
/nfs/users/ext_cvgroup-9/miniconda3/envs/cv703/bin/python -m torch.distributed.launch --nproc_per_node=8 --use_env  main.py\
 --isaid_path /nfs/projects/cv703/jazz-cvgroup-9 \
 --output_dir ./Outputs/${Name}_${backbone}_ep${epochs}_q${queries}_lr${lr}_${lr_backbone} \
 --backbone ${backbone} \
 --lr ${lr} \
 --lr_backbone ${lr_backbone} \
 --num_queries ${queries} > Outs/${Name}_${backbone}_ep${epochs}_q${queries}_lr${lr}_${lr_backbone}.out


# watch -n 5 nvidia-smi


# /nfs/users/ext_cvgroup-9/miniconda3/envs/cv703/bin/python -m torch.distributed.launch --nproc_per_node=8 --use_env  main.py\
#  --isaid_path /nfs/projects/cv703/jazz-cvgroup-9 \
#  --output_dir ./Outputs/${Name}_${backbone}_ep${epochs}_q${queries}_lr${lr}_${lr_backbone} \
#  --backbone ${backbone} \
#  --lr ${lr} \
#  --lr_backbone ${lr_backbone} \
#  --num_queries ${queries} > Outs/${Name}_${backbone}_ep${epochs}_q${queries}_lr${lr}_${lr_backbone}.out