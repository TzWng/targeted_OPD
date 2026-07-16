# -*- coding: utf-8 -*-
"""数据加载与提示词构造 —— 与训练脚本一致(SYS_MATH / SYS_GSM8K + chat template)。"""
import random

SYS_GSM8K = ("You are a helpful math assistant. Solve the problem step by step, "
             "then give the final answer on a new line as '#### <number>'.")
SYS_MATH = ("Please reason step by step, and put your final answer within "
            "\\boxed{}.")


def build_prompt(cfg, tok, question: str) -> str:
    sys = SYS_GSM8K if cfg.dataset == "gsm8k" else SYS_MATH
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_candidates(cfg):
    """返回打乱后的前 screen_candidates 道候选题 [{question, gold}]。"""
    from datasets import load_dataset
    if cfg.dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")["train"].shuffle(seed=cfg.seed)
        items = [{"question": ex["question"],
                  "gold": ex["answer"].split("####")[-1].strip().replace(",", "")}
                 for ex in ds.select(range(cfg.screen_candidates))]
    elif cfg.dataset == "deepmath":
        tr = load_dataset("zwhe99/DeepMath-103K", split="train")
        tr = tr.filter(lambda ex: cfg.dm_diff_min <= ex["difficulty"] <= cfg.dm_diff_max)
        tr = tr.shuffle(seed=cfg.seed)
        print(f"[data] DeepMath 难度带 [{cfg.dm_diff_min}, {cfg.dm_diff_max}]: {len(tr)} 题")
        items = [{"question": ex["question"], "gold": ex["final_answer"]}
                 for ex in tr.select(range(min(cfg.screen_candidates, len(tr))))]
    else:
        raise ValueError(f"未知数据集 {cfg.dataset}")
    return items


def select_probe_questions(cfg, screened):
    """screened: [{question, gold, pass_rate}] → 选 n_questions 道、通过率最接近 0.5 的混合题。"""
    inside = [s for s in screened if cfg.pass_lo <= s["pass_rate"] <= cfg.pass_hi]
    pool = inside if len(inside) >= cfg.n_questions else screened
    if len(inside) < cfg.n_questions:
        print(f"[screen] 警告:窗口 [{cfg.pass_lo},{cfg.pass_hi}] 内只有 {len(inside)} 题,"
              f"退而取全体中最接近 0.5 的 {cfg.n_questions} 道")
    pool = sorted(pool, key=lambda s: abs(s["pass_rate"] - 0.5))
    picked = pool[:cfg.n_questions]
    rng = random.Random(cfg.seed)
    rng.shuffle(picked)
    return picked
