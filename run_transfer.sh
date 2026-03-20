#!/bin/bash
#SBATCH --job-name=speck-xfer
#SBATCH --output=%j_out.log
#SBATCH --error=%j_error.log
#SBATCH --time=01:00:00
#SBATCH --qos=short
#SBATCH --partition=short
#SBATCH --mem=16G
#SBATCH --account=ravi
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=gpu01
#SBATCH --gres=gpu:3g.40gb:1

echo "Job started on $(hostname) at $(date)"

source ~/miniconda3/bin/activate cryptanalysis

python train_transfer.py

echo "Job finished at $(date)"
