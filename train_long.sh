#!/bin/bash
#SBATCH --job-name=deter_tome
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --cpus-per-task=40
#SBATCH --partition=multigpu

# /nfs/users/ext_cvgroup-9/miniconda3/envs/cv703/bin/python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --isaid_path /nfs/projects/cv703/jazz-cvgroup-9 --output_dir ./Outputs/TOME_encoder_128_batch1/  > Outs/TOME_encoder_128_batch1.out
CUDA_VISIBLE_DEVICES=4 /nfs/users/ext_cvgroup-9/miniconda3/envs/cv703/bin/python  main.py --isaid_path /nfs/projects/cv703/jazz-cvgroup-9 --output_dir ./Outputs/TOME_DETR_32_resume_frm_trained_baseline/ --resume /nfs/users/ext_sanoojan.baliah/Sanoojan/detr/Outputs/baseline/checkpoint0299.pth > Outs/TOME_DETR_32_resume_frm_trained_baseline.out


