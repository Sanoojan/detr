#!/bin/sh
#SBATCH --job-name=detr_baseline
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=10
#SBATCH --partition=default-short

#cd ../..

# /nfs/users/ext_cvgroup-9/miniconda3/envs/cv703/bin/python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --isaid_path /nfs/projects/cv703/jazz-cvgroup-9 --output_dir ./Outputs/TOME_encoder_128_batch1/  > Outs/TOME_encoder_128_batch1.out
# /nfs/users/ext_cvgroup-9/miniconda3/envs/cv703/bin/python  main.py --isaid_path /nfs/projects/cv703/jazz-cvgroup-9 --output_dir ./Outputs/Eval/ --resume /nfs/users/ext_sanoojan.baliah/Sanoojan/detr/Outputs/baseline/checkpoint0299.pth > Outs/TOME_DETR_32_resume_frm_trained_baseline.out

/nfs/users/ext_cvgroup-9/miniconda3/envs/cv703/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --batch_size 2 --no_aux_loss --eval --resume /nfs/users/ext_sanoojan.baliah/Sanoojan/detr/Outputs/baseline_continue/checkpoint0899.pth --isaid_path /nfs/projects/cv703/jazz-cvgroup-9 --output_dir ./Outputs/Eval_baseline8/ > Outs/eval_baseline_899.out
/nfs/users/ext_cvgroup-9/miniconda3/envs/cv703/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --batch_size 2 --no_aux_loss --eval --resume /nfs/users/ext_sanoojan.baliah/Sanoojan/detr/Outputs/baseline_continue/checkpoint0799.pth --isaid_path /nfs/projects/cv703/jazz-cvgroup-9 --output_dir ./Outputs/Eval_baseline7/ > Outs/eval_baseline_799.out
/nfs/users/ext_cvgroup-9/miniconda3/envs/cv703/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --batch_size 2 --no_aux_loss --eval --resume /nfs/users/ext_sanoojan.baliah/Sanoojan/detr/Outputs/baseline_continue/checkpoint0699.pth --isaid_path /nfs/projects/cv703/jazz-cvgroup-9 --output_dir ./Outputs/Eval_baseline6/ > Outs/eval_baseline_699.out
/nfs/users/ext_cvgroup-9/miniconda3/envs/cv703/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --batch_size 2 --no_aux_loss --eval --resume /nfs/users/ext_sanoojan.baliah/Sanoojan/detr/Outputs/baseline_continue/checkpoint0599.pth --isaid_path /nfs/projects/cv703/jazz-cvgroup-9 --output_dir ./Outputs/Eval_baseline5/ > Outs/eval_baseline_599.out
/nfs/users/ext_cvgroup-9/miniconda3/envs/cv703/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --batch_size 2 --no_aux_loss --eval --resume /nfs/users/ext_sanoojan.baliah/Sanoojan/detr/Outputs/baseline_continue/checkpoint0499.pth --isaid_path /nfs/projects/cv703/jazz-cvgroup-9 --output_dir ./Outputs/Eval_baseline4/ > Outs/eval_baseline_499.out
/nfs/users/ext_cvgroup-9/miniconda3/envs/cv703/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --batch_size 2 --no_aux_loss --eval --resume /nfs/users/ext_sanoojan.baliah/Sanoojan/detr/Outputs/baseline_continue/checkpoint0399.pth --isaid_path /nfs/projects/cv703/jazz-cvgroup-9 --output_dir ./Outputs/Eval_baseline3/ > Outs/eval_baseline_399.out
/nfs/users/ext_cvgroup-9/miniconda3/envs/cv703/bin/python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --batch_size 1 --no_aux_loss --eval --resume /nfs/users/ext_sanoojan.baliah/Sanoojan/detr/Outputs/baseline_continue/checkpoint0999.pth --isaid_path /nfs/projects/cv703/jazz-cvgroup-9 --output_dir ./Outputs/Eval_baseline9_batch1/ > Outs/eval_baseline_999_1.out
