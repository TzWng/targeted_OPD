# -*- coding: utf-8 -*-
"""实验配置。运行开始时会把解析后的完整配置打印并存盘(manifest),
避免 exp-0715 里"日志头与脚本默认值对不上"的问题(诊断 F9)。"""
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass
class ProbeConfig:
    # ---- 模型 ----
    student: str = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path: Optional[str] = None   # 可选:LoRA 检查点目录(探训练中期策略时用)
    teacher: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    dtype: str = "auto"                  # auto | bf16 | fp16

    # ---- 数据 ----
    dataset: str = "deepmath"            # deepmath | gsm8k(与训练战役一致)
    dm_diff_min: float = 0.0
    dm_diff_max: float = 3.0

    # ---- 题目筛选(保证探针题有对错混合,否则参考陪审团退化) ----
    screen_candidates: int = 64          # 先粗筛的候选题数
    screen_rollouts: int = 16            # 每道候选题的粗筛采样数
    pass_lo: float = 0.15                # 入选的通过率窗口(取最接近 0.5 的 n_questions 道)
    pass_hi: float = 0.85

    # ---- 主采样 ----
    n_questions: int = 8                 # 探针题数(计划 D2)
    rollouts_per_question: int = 256     # 参考陪审团大小 R
    max_new_tokens: int = 1024           # 放宽的生成上限(训练时 512/768;可再调到 2048)
    max_prompt_tokens: int = 512
    temperature: float = 1.0             # 必须 1.0:采样分布 = 被分析的策略(讨论纪要 §3)
    top_p: float = 1.0
    include_truncated: bool = True       # 忠实于训练(截断样本也计入);上限已放宽,占比应很低

    # ---- D2:陪审团大小扫描 ----
    jury_sizes: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    juries_per_size: int = 32            # 每个大小重采样的陪审团个数(每道题)

    # ---- D3:三种票型(组大小固定为训练的 G) ----
    d3_group_size: int = 8
    d3_groups_per_question: int = 25     # 8 题 × 25 = 200 个模拟训练组

    # ---- 计算 ----
    forward_batch: int = 4               # 每次 HF 前向的 rollout 数
    influence_chunk: int = 128           # G 累加/打分时的 token 分块
    seed: int = 0
    out_dir: str = "outputs"
    vllm_gpu_frac: float = 0.85          # 采样阶段独占 GPU,可以给高

    def resolve_dtype(self):
        import torch
        if self.dtype == "bf16":
            return torch.bfloat16
        if self.dtype == "fp16":
            return torch.float16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def dump(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=1)


def load_config(yaml_path: Optional[str] = None, **overrides) -> ProbeConfig:
    cfg = ProbeConfig()
    if yaml_path:
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if not hasattr(cfg, k):
                raise KeyError(f"config.yaml 里有未知字段: {k}")
            setattr(cfg, k, v)
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def apply_smoke(cfg: ProbeConfig) -> ProbeConfig:
    """--smoke:10 分钟级端到端冒烟(诊断计划的'先干跑'护栏)。"""
    cfg.screen_candidates = 8
    cfg.screen_rollouts = 8
    cfg.n_questions = 2
    cfg.rollouts_per_question = 32
    cfg.max_new_tokens = 512
    cfg.jury_sizes = [4, 8]
    cfg.juries_per_size = 4
    cfg.d3_group_size = 8
    cfg.d3_groups_per_question = 4
    return cfg
