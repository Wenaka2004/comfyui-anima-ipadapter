#!/bin/bash
#SBATCH --job-name=ipa-train
#SBATCH --partition=standard
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --output=/data/stardust/anima_ipa/stage7_train/logs/train_ipa_%j.out

cd /data/stardust/anima_ipa/stage7_train/code

export WANDB_API_KEY="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"
export CUDA_VISIBLE_DEVICES=0,1

/home/stardust/miniconda3/envs/comfyui/bin/python train_ipa_kohya.py \
    --wandb \
    --num-steps 100000 \
    --batch-size 8 \
    --lr 1e-4 \
    --save-every 1000 \
    --log-every 10 \
    --output-dir /data/stardust/anima_ipa/stage7_train/out_ipa
