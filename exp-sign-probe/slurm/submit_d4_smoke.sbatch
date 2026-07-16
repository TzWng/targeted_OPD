#!/bin/bash
#SBATCH --job-name=opd-d4-smoke
#SBATCH --partition=gpu-redhat
#SBATCH --constraint=ampere
#SBATCH --exclude=gpu018
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -e
export PATH=/scratch/$USER/envs/opd-probe/bin:$PATH
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_home}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
echo "== D4 单元测试(计算节点)=="
python tests/test_d4_small.py
python -m probe.d4_projection --run-dir /scratch/$USER/opd_sign_probe/run1_smoke \
    --smoke --lora-r 32 --lora-alpha 64
echo "===== D4 冒烟报告 ====="
cat /scratch/$USER/opd_sign_probe/run1_smoke/d4/d4_report.md
