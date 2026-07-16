#!/bin/bash
# 在 Amarel 登录节点运行(计算节点通常无外网):把模型与数据集下载到 scratch。
set -e
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_home}
echo "HF_HOME=$HF_HOME"
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-Math-1.5B-Instruct
python - <<'PY'
from datasets import load_dataset
load_dataset("zwhe99/DeepMath-103K", split="train")
load_dataset("openai/gsm8k", "main")
print("datasets cached OK")
PY
echo "全部下载完成 → $HF_HOME"
