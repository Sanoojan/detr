#!/bin/sh
#SBATCH --job-name=tome
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=10
#SBATCH --partition=default-short

#cd ../..

# /nfs/users/ext_cvgroup-9/miniconda3/envs/cv703/bin/python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --isaid_path /nfs/projects/cv703/jazz-cvgroup-9 --output_dir ./Outputs/TOME_encoder_128_batch1/  > Outs/TOME_encoder_128_batch1.out
# /nfs/users/ext_cvgroup-9/miniconda3/envs/cv703/bin/python  main.py --isaid_path /nfs/projects/cv703/jazz-cvgroup-9 --output_dir ./Outputs/Eval/ --resume /nfs/users/ext_sanoojan.baliah/Sanoojan/detr/Outputs/baseline/checkpoint0299.pth > Outs/TOME_DETR_32_resume_frm_trained_baseline.out
fuser -k 29500/tcp

/nfs/users/ext_cvgroup-9/miniconda3/envs/cv703/bin/python main.py \
    --batch_size 2 \
    --no_aux_loss \
    --eval \
    --resume /nfs/users/ext_sanoojan.baliah/Sanoojan/detr/Outputs/baseline/checkpoint0299.pth \
    --isaid_path /nfs/projects/cv703/jazz-cvgroup-9 \
    --output_dir ./Outputs/Eval/TOME_no > Out_eval/TOME_no.out

watch nvidia-smi