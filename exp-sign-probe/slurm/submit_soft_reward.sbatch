#!/bin/bash
#SBATCH --job-name=opd-soft-reward
#SBATCH --partition=main-redhat
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -e
export PATH=/scratch/$USER/envs/opd-probe/bin:$PATH PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
python -m probe.soft_reward_analysis --run-dir /scratch/$USER/opd_sign_probe/run1
python -m probe.soft_reward_analysis --run-dir /scratch/$USER/opd_sign_probe/run2_gsm8k
echo "===== DeepMath ====="; cat /scratch/$USER/opd_sign_probe/run1/results/soft_reward_report.md
echo "===== GSM8K ====="; cat /scratch/$USER/opd_sign_probe/run2_gsm8k/results/soft_reward_report.md
