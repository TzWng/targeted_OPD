#!/bin/bash
# 建 conda 环境(登录节点运行一次)。如已有 miniconda/miniforge 可跳过安装段。
set -e
ENV_PREFIX=/scratch/$USER/envs/opd-probe   # 放 scratch,避免 home 配额
source "$(conda info --base)/etc/profile.d/conda.sh"
[ -d "$ENV_PREFIX" ] || conda create -y -p "$ENV_PREFIX" python=3.11
conda activate "$ENV_PREFIX"
export PIP_CACHE_DIR=/scratch/$USER/.pip-cache TMPDIR=/scratch/$USER/tmp
mkdir -p $PIP_CACHE_DIR $TMPDIR
# vllm 会自带匹配的 torch;math-verify 是训练同款判分器
pip install "vllm~=0.10.0" "transformers>=4.44" peft datasets accelerate math-verify numpy pyyaml
python -c "import torch, vllm, transformers; print('torch', torch.__version__, '| vllm', vllm.__version__)"
echo "环境 $ENV_PREFIX 就绪。"
