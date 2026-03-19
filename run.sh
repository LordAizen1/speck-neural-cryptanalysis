#!/bin/bash
#SBATCH --job-name=speck-crypto
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

# Activate conda environment
source ~/miniconda3/bin/activate cryptanalysis

# Verify GPU
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Run training
python train.py

echo "Job finished at $(date)"
