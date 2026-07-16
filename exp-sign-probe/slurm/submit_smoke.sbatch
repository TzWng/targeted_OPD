#!/bin/bash
#SBATCH --job-name=opd-probe-smoke
#SBATCH --partition=gpu-redhat     # Amarel 主 GPU 分区(2026-07 实测)
#SBATCH --constraint=ampere        # A100:保证 bf16;排队久可删
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -e
export PATH=/scratch/$USER/envs/opd-probe/bin:$PATH   # 直接用环境二进制,不依赖节点上的 conda
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_home}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
OUT=/scratch/$USER/opd_sign_probe/run1     # --smoke 会自动写到 ${OUT}_smoke
python -m probe.run_probe --config config.yaml --out $OUT --stage all --smoke
echo "===== 冒烟报告 ====="
cat "${OUT}_smoke/results/report.md"
