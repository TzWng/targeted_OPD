#!/bin/bash
#SBATCH --job-name=opd-d4-gsm8k
#SBATCH --partition=gpu-redhat
#SBATCH --constraint=ampere
#SBATCH --exclude=gpu018
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -e
export PATH=/scratch/$USER/envs/opd-probe/bin:$PATH
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_home}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
python -m probe.d4_projection --run-dir /scratch/$USER/opd_sign_probe/run2_gsm8k --lora-r 16 --lora-alpha 32
echo "===== D4 冒烟报告 ====="
cat /scratch/$USER/opd_sign_probe/run2_gsm8k/d4/d4_report.md
