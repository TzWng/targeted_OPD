#!/bin/bash
#SBATCH --job-name=opd-block-analysis
#SBATCH --partition=main-redhat    # 纯 CPU 分析;登录节点 glibc 太老,env 只能在计算节点跑
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -e
export PATH=/scratch/$USER/envs/opd-probe/bin:$PATH   # 直接用环境二进制,不依赖节点上的 conda
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_home}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
python tests/test_block_analysis_small.py             # 暴力对照单测先行
python -m probe.block_analysis --config config.yaml       --out /scratch/$USER/opd_sign_probe/run1
python -m probe.block_analysis --config config_gsm8k.yaml --out /scratch/$USER/opd_sign_probe/run2_gsm8k
echo "===== DeepMath ====="; cat /scratch/$USER/opd_sign_probe/run1/results/block_report.md
echo "===== GSM8K =====";    cat /scratch/$USER/opd_sign_probe/run2_gsm8k/results/block_report.md
