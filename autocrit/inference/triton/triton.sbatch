#!/bin/bash
#SBATCH -p g40
#SBATCH --account trlx
#SBATCH --gres=gpu:1
#SBATCH --output="%x.out"
#SBATCH --job-name=triton

srun --container-image nvcr.io#nvidia/tritonserver:23.01-py3 --container-mounts=/fsx/home-uwu/autocrit/autocrit/inference/model_store:/models tritonserver --model-repository=/models
