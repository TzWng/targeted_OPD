# -*- coding: utf-8 -*-
"""编排入口。
用法:
  python -m probe.run_probe --config config.yaml --out /scratch/$USER/opd_sign_probe/run1 \
      --stage all [--smoke] [--adapter PATH] [--seed 0]
阶段:rollout(采样,vLLM)→ teacher → reference → d2 → d3 → analyze;
各阶段有产物即跳过(断点续跑);建议 sbatch 里按阶段分开调用以彻底释放显存。"""
import argparse
import gc
import json
import os
import random
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stage", default="all",
                    choices=["rollout", "teacher", "reference", "d2", "d3",
                             "analyze", "all"])
    ap.add_argument("--smoke", action="store_true", help="10 分钟级端到端冒烟")
    ap.add_argument("--adapter", default=None, help="LoRA 检查点目录(可选)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    from .config import apply_smoke, load_config
    cfg = load_config(args.config, out_dir=args.out, adapter_path=args.adapter,
                      seed=args.seed)
    if args.smoke:
        cfg = apply_smoke(cfg)
        cfg.out_dir = cfg.out_dir.rstrip("/") + "_smoke"

    os.makedirs(cfg.out_dir, exist_ok=True)
    manifest = os.path.join(cfg.out_dir, "manifest.json")
    cfg.dump(manifest)
    print(f"[config] 解析后配置已存 {manifest}:")
    print(json.dumps(json.load(open(manifest)), ensure_ascii=False, indent=1))
    _log_versions(cfg.out_dir)

    import numpy as np
    import torch
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    stages = ([args.stage] if args.stage != "all"
              else ["rollout", "teacher", "reference", "d2", "d3", "analyze"])
    for st in stages:
        print(f"\n===== 阶段: {st} =====")
        if st == "rollout":
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(cfg.student)
            from .rollout import run_rollout_stage
            run_rollout_stage(cfg, tok)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif st == "teacher":
            from .stages import run_teacher_stage
            run_teacher_stage(cfg)
        elif st == "reference":
            from .stages import run_reference_stage
            run_reference_stage(cfg)
        elif st == "d2":
            from .stages import run_d2_stage
            run_d2_stage(cfg)
        elif st == "d3":
            from .stages import run_d3_stage
            run_d3_stage(cfg)
        elif st == "analyze":
            from .analysis import run_analysis
            run_analysis(cfg)
    print("\n[done] 全部请求阶段完成。结果目录:", cfg.out_dir)


def _log_versions(out_dir):
    info = {"python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "vllm", "peft", "datasets", "numpy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "n/a"
    with open(os.path.join(out_dir, "versions.json"), "w") as f:
        json.dump(info, f, indent=1)
    print("[versions]", info)


if __name__ == "__main__":
    main()
