#!/bin/bash
#SBATCH --job-name=opd-sign-probe
#SBATCH --partition=gpu-redhat     # Amarel 主 GPU 分区(2026-07 实测)
#SBATCH --constraint=ampere        # A100:保证 bf16;排队久可删
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -e
export PATH=/scratch/$USER/envs/opd-probe/bin:$PATH   # 直接用环境二进制,不依赖节点上的 conda
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_home}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
OUT=/scratch/$USER/opd_sign_probe/run1
# 分阶段调用:每阶段独立进程,彻底释放显存;有产物即跳过,可安全重交续跑
python -m probe.run_probe --config config.yaml --out $OUT --stage rollout
python -m probe.run_probe --config config.yaml --out $OUT --stage teacher
python -m probe.run_probe --config config.yaml --out $OUT --stage reference
python -m probe.run_probe --config config.yaml --out $OUT --stage d2
python -m probe.run_probe --config config.yaml --out $OUT --stage d3
python -m probe.run_probe --config config.yaml --out $OUT --stage analyze
echo "===== 报告 ====="; cat $OUT/results/report.md
