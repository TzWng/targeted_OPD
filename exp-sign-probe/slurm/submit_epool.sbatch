#!/bin/bash
#SBATCH --job-name=opd-epool-head
#SBATCH --partition=gpu-redhat
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -e
export PATH=/scratch/$USER/envs/opd-probe/bin:$PATH
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_home}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
python tests/test_e_pool_small.py                     # 暴力对照单测先行
for PAIR in "config.yaml /scratch/$USER/opd_sign_probe/run1" \
            "config_gsm8k.yaml /scratch/$USER/opd_sign_probe/run2_gsm8k"; do
  set -- $PAIR
  python -m probe.e_pool_head --config $1 --out $2 --stage cache
  python -m probe.e_pool_head --config $1 --out $2 --stage e1
  python -m probe.e_pool_head --config $1 --out $2 --stage e2
  python -m probe.e_pool_head --config $1 --out $2 --stage report
done
echo "===== DeepMath ====="; cat /scratch/$USER/opd_sign_probe/run1/results/e_report.md
echo "===== GSM8K =====";    cat /scratch/$USER/opd_sign_probe/run2_gsm8k/results/e_report.md
