# -*- coding: utf-8 -*-
"""阶段 1:vLLM 采样(粗筛 + 主采样)。
输出(out_dir 下):
  screen.json                     每道候选题的通过率
  questions.json                  选中的探针题
  q{idx}/meta.json                题目、gold、prompt_token_ids、通过率
  q{idx}/rollouts.jsonl           每行一条 rollout:completion_ids / verdict / truncated / text
温度必须 1.0(采样分布 = 被分析的策略);生成上限已放宽(默认 1024)。"""
import json
import os

from .data import build_prompt, load_candidates, select_probe_questions
from .verifier import extract_pred, verify


def _make_llm(cfg):
    from vllm import LLM
    kwargs = dict(
        model=cfg.student,
        gpu_memory_utilization=cfg.vllm_gpu_frac,
        max_model_len=cfg.max_prompt_tokens + cfg.max_new_tokens,
        seed=cfg.seed,
    )
    if cfg.adapter_path:
        kwargs.update(enable_lora=True, max_lora_rank=64)
    return LLM(**kwargs)


def _lora_request(cfg):
    if not cfg.adapter_path:
        return None
    from vllm.lora.request import LoRARequest
    return LoRARequest("probe_adapter", 1, cfg.adapter_path)


def _sampling(cfg, n, seed):
    from vllm import SamplingParams
    return SamplingParams(
        n=n, temperature=cfg.temperature, top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
        truncate_prompt_tokens=cfg.max_prompt_tokens, seed=seed,
    )


def _generate(llm, cfg, prompts, n, seed):
    req = _lora_request(cfg)
    sp = _sampling(cfg, n, seed)
    if req is not None:
        return llm.generate(prompts, sp, lora_request=req)
    return llm.generate(prompts, sp)


def run_rollout_stage(cfg, tok):
    os.makedirs(cfg.out_dir, exist_ok=True)
    qfile = os.path.join(cfg.out_dir, "questions.json")
    if os.path.exists(qfile):
        print("[rollout] questions.json 已存在,跳过(如需重跑请删除输出目录)")
        return

    llm = _make_llm(cfg)

    # ---- 粗筛:candidates × screen_rollouts ----
    cands = load_candidates(cfg)
    prompts = [build_prompt(cfg, tok, c["question"]) for c in cands]
    outs = _generate(llm, cfg, prompts, cfg.screen_rollouts, cfg.seed)
    screened = []
    for c, out in zip(cands, outs):
        vs = [verify(extract_pred(o.text), c["gold"]) for o in out.outputs]
        screened.append({**c, "pass_rate": sum(vs) / len(vs)})
    with open(os.path.join(cfg.out_dir, "screen.json"), "w") as f:
        json.dump(screened, f, ensure_ascii=False, indent=1)
    rates = sorted(s["pass_rate"] for s in screened)
    print(f"[screen] {len(screened)} 题通过率分布: min={rates[0]:.2f} "
          f"med={rates[len(rates)//2]:.2f} max={rates[-1]:.2f}")

    picked = select_probe_questions(cfg, screened)
    print("[screen] 选中探针题通过率:", [round(p["pass_rate"], 2) for p in picked])

    # ---- 主采样:每题 R 条 ----
    prompts = [build_prompt(cfg, tok, p["question"]) for p in picked]
    outs = _generate(llm, cfg, prompts, cfg.rollouts_per_question, cfg.seed + 1)
    qmeta = []
    for qi, (p, out) in enumerate(zip(picked, outs)):
        qdir = os.path.join(cfg.out_dir, f"q{qi}")
        os.makedirs(qdir, exist_ok=True)
        n_trunc, n_pass = 0, 0
        with open(os.path.join(qdir, "rollouts.jsonl"), "w") as f:
            for o in out.outputs:
                v = verify(extract_pred(o.text), p["gold"])
                trunc = (o.finish_reason == "length")
                n_trunc += int(trunc)
                n_pass += int(v > 0)
                f.write(json.dumps({
                    "completion_ids": list(o.token_ids),
                    "verdict": int(v), "truncated": bool(trunc), "text": o.text,
                }, ensure_ascii=False) + "\n")
        meta = {"idx": qi, "question": p["question"], "gold": p["gold"],
                "screen_pass_rate": p["pass_rate"],
                "prompt_token_ids": list(out.prompt_token_ids),
                "R": len(out.outputs),
                "main_pass_rate": n_pass / len(out.outputs),
                "trunc_rate": n_trunc / len(out.outputs)}
        with open(os.path.join(qdir, "meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        qmeta.append(meta)
        print(f"[rollout] q{qi}: pass={meta['main_pass_rate']:.2f} "
              f"trunc={meta['trunc_rate']:.2f} (cap={cfg.max_new_tokens})")
    with open(qfile, "w") as f:
        json.dump(qmeta, f, ensure_ascii=False, indent=1)
    print(f"[rollout] 完成:{len(qmeta)} 题 × {cfg.rollouts_per_question} 条 → {cfg.out_dir}")


def load_question(cfg, qi):
    """读回一道题的 rollouts:meta, [{completion_ids, verdict, truncated}]。"""
    qdir = os.path.join(cfg.out_dir, f"q{qi}")
    meta = json.load(open(os.path.join(qdir, "meta.json")))
    rolls = [json.loads(line) for line in open(os.path.join(qdir, "rollouts.jsonl"))]
    if not cfg.include_truncated:
        rolls = [r for r in rolls if not r["truncated"]]
    return meta, rolls
