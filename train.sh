

# /nfs/users/ext_cvgroup-9/miniconda3/envs/cv703/bin/python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --isaid_path /nfs/projects/cv703/jazz-cvgroup-9 --output_dir ./Outputs/baseline_continue/  --resume /nfs/users/ext_sanoojan.baliah/Sanoojan/detr/Outputs/baseline/checkpoint0299.pth> Outs/baseline_batch2.out
/nfs/users/ext_cvgroup-9/miniconda3/envs/cv703/bin/python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --isaid_path /nfs/projects/cv703/jazz-cvgroup-9 --output_dir ./Outputs/TOME_encoder_128_batch1/  > Outs/TOME_encoder_128_batch1.out
watch -n 5 nvidia-smi