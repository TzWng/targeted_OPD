# -*- coding: utf-8 -*-
"""阶段 2–4:参考真值(R=256 大陪审团)、D2 陪审团扫描、D3 三票型。
全部忠实于训练实现的约定:
  * 优势 a_i = V_i − 陪审团均值(含自身;留一只从 G 里减去自身贡献);
  * 分半按成员列表位置的奇偶(训练代码 (bi % G) % 2 的组内奇偶);
  * 半批 G 用全陪审团的优势(训练代码 G_half 的做法);
  * 对面半批打分不做留一(对面半批天然不含自己)。"""
import gc
import json
import os
import re

import numpy as np
import torch

from .influence import (build_jury_G, iota_from_G, iter_forward,
                        own_correction, teacher_logp)
from .rollout import load_question

_SYM = re.compile(r"[+\-*/=<>^\\(){}\[\]_%$|.,:;!?~&#]")
_DIG = re.compile(r"\d")
_ALP = re.compile(r"[A-Za-z]")
TOKEN_TYPES = ["digit", "symbol", "word", "other"]


def token_type(tok, tid):
    s = tok.decode([int(tid)])
    if _DIG.search(s):
        return 0
    if _SYM.search(s):
        return 1
    if _ALP.search(s):
        return 2
    return 3


def load_student(cfg, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.student)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.student, torch_dtype=cfg.resolve_dtype(), attn_implementation="sdpa"
    ).to(device)
    if cfg.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg.adapter_path)
        print(f"[model] 已加载 LoRA 适配器: {cfg.adapter_path}")
    model.eval()
    return tok, model


def rollout_tensors(meta, rolls):
    prompt = torch.tensor(meta["prompt_token_ids"], dtype=torch.long)
    out = []
    for r in rolls:
        comp = torch.tensor(r["completion_ids"], dtype=torch.long)
        out.append({"ids": torch.cat([prompt, comp]), "plen": int(prompt.shape[0]),
                    "T": int(comp.shape[0])})
    return out


def _question_ids(cfg):
    qfile = os.path.join(cfg.out_dir, "questions.json")
    return [m["idx"] for m in json.load(open(qfile))]


def _adv_from(verdicts, members):
    v = verdicts[members]
    return None if v.min() == v.max() else v - v.mean()   # None ⇒ 退化陪审团


# ================================================================ 参考真值
def run_teacher_stage(cfg, device="cuda"):
    from transformers import AutoModelForCausalLM
    done = all(os.path.exists(os.path.join(cfg.out_dir, f"q{qi}", "teacher_logp.npz"))
               for qi in _question_ids(cfg))
    if done:
        print("[teacher] 已全部存在,跳过")
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.student)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher, torch_dtype=cfg.resolve_dtype(), attn_implementation="sdpa"
    ).to(device)
    teacher.eval()
    for qi in _question_ids(cfg):
        path = os.path.join(cfg.out_dir, f"q{qi}", "teacher_logp.npz")
        if os.path.exists(path):
            continue
        meta, rolls = load_question(cfg, qi)
        rts = rollout_tensors(meta, rolls)
        tl = teacher_logp(teacher, tok.pad_token_id, rts, device, cfg.forward_batch)
        np.savez_compressed(path, **{f"r{i}": t.float().numpy() for i, t in enumerate(tl)})
        print(f"[teacher] q{qi} 完成 ({len(rts)} 条)")
    del teacher
    gc.collect()
    torch.cuda.empty_cache()


def run_reference_stage(cfg, device="cuda"):
    tok, student = load_student(cfg, device)
    V = student.config.vocab_size
    hd = student.config.hidden_size
    pad = tok.pad_token_id
    for qi in _question_ids(cfg):
        out_path = os.path.join(cfg.out_dir, f"q{qi}", "tokens.npz")
        if os.path.exists(out_path):
            print(f"[reference] q{qi} 已存在,跳过")
            continue
        meta, rolls = load_question(cfg, qi)
        rts = rollout_tensors(meta, rolls)
        R = len(rts)
        verdicts = np.array([r["verdict"] for r in rolls], dtype=np.float32)
        adv = verdicts - verdicts.mean()
        tl = np.load(os.path.join(cfg.out_dir, f"q{qi}", "teacher_logp.npz"))

        # 参考陪审团 = 全部 R 条;同时按奇偶建两半(参考自稳定性检验用)
        G_ref, G_A, G_B = build_jury_G(student, pad, rts, adv, list(range(R)),
                                       device, V, hd, cfg.forward_batch, halves=True)
        cols = {k: [] for k in ["rollout", "pos", "y", "d", "logp_s", "ref_iota",
                                "signA", "signB", "ttype", "trunc", "verdict"]}
        for local, probs, h, y, logp in iter_forward(student, pad, rts, device,
                                                     cfg.forward_batch):
            if probs is None:
                continue
            corr = own_correction(probs, h, y)
            a_j = float(adv[local])
            ref = iota_from_G(G_ref, probs, h, y) - a_j * corr
            iA = iota_from_G(G_A, probs, h, y)
            iB = iota_from_G(G_B, probs, h, y)
            if local % 2 == 0:
                iA = iA - a_j * corr        # 自己在 A 半,A 票做留一
            else:
                iB = iB - a_j * corr
            T = int(y.shape[0])
            d = torch.from_numpy(tl[f"r{local}"]).to(device) - logp
            cols["rollout"].append(np.full(T, local, np.int32))
            cols["pos"].append(np.arange(T, dtype=np.int32))
            cols["y"].append(y.cpu().numpy().astype(np.int32))
            cols["d"].append(d.float().cpu().numpy())
            cols["logp_s"].append(logp.float().cpu().numpy())
            cols["ref_iota"].append(ref.float().cpu().numpy())
            cols["signA"].append(np.sign(iA.float().cpu().numpy()).astype(np.int8))
            cols["signB"].append(np.sign(iB.float().cpu().numpy()).astype(np.int8))
            cols["ttype"].append(np.array([token_type(tok, t) for t in y.cpu()],
                                          dtype=np.int8))
            cols["trunc"].append(np.full(T, int(rolls[local]["truncated"]), np.int8))
            cols["verdict"].append(np.full(T, int(rolls[local]["verdict"]), np.int8))
            del probs, h
        np.savez_compressed(out_path, **{k: np.concatenate(v) for k, v in cols.items()})
        del G_ref, G_A, G_B
        torch.cuda.empty_cache()
        print(f"[reference] q{qi} 完成:{R} 条 rollout,"
              f"{sum(len(x) for x in cols['pos'])} 个 token")
    del student
    gc.collect()
    torch.cuda.empty_cache()


# ================================================================ D2:陪审团扫描
def run_d2_stage(cfg, device="cuda"):
    tok, student = load_student(cfg, device)
    V, hd, pad = student.config.vocab_size, student.config.hidden_size, tok.pad_token_id
    for qi in _question_ids(cfg):
        out_path = os.path.join(cfg.out_dir, f"q{qi}", "d2.npz")
        if os.path.exists(out_path):
            print(f"[d2] q{qi} 已存在,跳过")
            continue
        meta, rolls = load_question(cfg, qi)
        rts = rollout_tensors(meta, rolls)
        R = len(rts)
        verdicts = np.array([r["verdict"] for r in rolls], dtype=np.float32)
        rng = np.random.default_rng(cfg.seed * 1000 + qi)
        rows = {k: [] for k in ["size", "jury", "rollout", "pos", "iota_hat", "alt_sign"]}
        jmeta = []
        for size in cfg.jury_sizes:
            for k in range(cfg.juries_per_size):
                members = rng.choice(R, size=size, replace=False).tolist()
                a = _adv_from(verdicts, np.array(members))
                if a is None:
                    jmeta.append({"size": size, "jury": k, "degenerate": True})
                    continue
                adv_full = np.zeros(R, np.float32)
                adv_full[np.array(members)] = a
                Gf, G0, G1 = build_jury_G(student, pad, rts, adv_full, members,
                                          device, V, hd, cfg.forward_batch, halves=True)
                mem_rts = [rts[i] for i in members]
                for local, probs, h, y, _ in iter_forward(student, pad, mem_rts,
                                                          device, cfg.forward_batch):
                    if probs is None:
                        continue
                    ri = members[local]
                    corr = own_correction(probs, h, y)
                    ih = (iota_from_G(Gf, probs, h, y)
                          - float(adv_full[ri]) * corr)
                    opp = G1 if (local % 2 == 0) else G0
                    alt = iota_from_G(opp, probs, h, y)
                    T = int(y.shape[0])
                    rows["size"].append(np.full(T, size, np.int16))
                    rows["jury"].append(np.full(T, k, np.int16))
                    rows["rollout"].append(np.full(T, ri, np.int32))
                    rows["pos"].append(np.arange(T, dtype=np.int32))
                    rows["iota_hat"].append(ih.float().cpu().numpy())
                    rows["alt_sign"].append(np.sign(alt.float().cpu().numpy()).astype(np.int8))
                    del probs, h
                jmeta.append({"size": size, "jury": k, "degenerate": False,
                              "pass_rate": float(verdicts[np.array(members)].mean())})
                del Gf, G0, G1
                torch.cuda.empty_cache()
        np.savez_compressed(out_path, **{k: np.concatenate(v) if v else np.array([])
                                         for k, v in rows.items()})
        json.dump(jmeta, open(os.path.join(cfg.out_dir, f"q{qi}", "d2_juries.json"), "w"))
        n_deg = sum(j["degenerate"] for j in jmeta)
        print(f"[d2] q{qi} 完成:{len(jmeta)} 个陪审团(退化 {n_deg})")
    del student
    gc.collect()
    torch.cuda.empty_cache()


# ================================================================ D3:三票型
def run_d3_stage(cfg, device="cuda"):
    tok, student = load_student(cfg, device)
    V, hd, pad = student.config.vocab_size, student.config.hidden_size, tok.pad_token_id
    for qi in _question_ids(cfg):
        out_path = os.path.join(cfg.out_dir, f"q{qi}", "d3.npz")
        if os.path.exists(out_path):
            print(f"[d3] q{qi} 已存在,跳过")
            continue
        meta, rolls = load_question(cfg, qi)
        rts = rollout_tensors(meta, rolls)
        R = len(rts)
        verdicts = np.array([r["verdict"] for r in rolls], dtype=np.float32)
        rng = np.random.default_rng(cfg.seed * 2000 + qi)
        rows = {k: [] for k in ["group", "rollout", "pos", "v1", "v2", "v3"]}
        n_deg = 0
        for g in range(cfg.d3_groups_per_question):
            members = rng.choice(R, size=cfg.d3_group_size, replace=False).tolist()
            a = _adv_from(verdicts, np.array(members))
            if a is None:
                n_deg += 1
                continue
            adv_full = np.zeros(R, np.float32)
            adv_full[np.array(members)] = a
            Gf, G0, G1 = build_jury_G(student, pad, rts, adv_full, members,
                                      device, V, hd, cfg.forward_batch, halves=True)
            mem_rts = [rts[i] for i in members]
            for local, probs, h, y, _ in iter_forward(student, pad, mem_rts,
                                                      device, cfg.forward_batch):
                if probs is None:
                    continue
                ri = members[local]
                a_j = float(adv_full[ri])
                corr = own_correction(probs, h, y)
                own, opp = (G0, G1) if (local % 2 == 0) else (G1, G0)
                v1 = iota_from_G(Gf, probs, h, y) - a_j * corr    # 票1:全组留一(训练的 iota)
                v2 = iota_from_G(opp, probs, h, y)                # 票2:对面半批(训练的 iota_alt)
                v3 = iota_from_G(own, probs, h, y) - a_j * corr   # 票3:本半批去己(真独立于票2)
                T = int(y.shape[0])
                rows["group"].append(np.full(T, g, np.int16))
                rows["rollout"].append(np.full(T, ri, np.int32))
                rows["pos"].append(np.arange(T, dtype=np.int32))
                for key, v in (("v1", v1), ("v2", v2), ("v3", v3)):
                    rows[key].append(v.float().cpu().numpy())
                del probs, h
            del Gf, G0, G1
            torch.cuda.empty_cache()
        np.savez_compressed(out_path, **{k: np.concatenate(v) if v else np.array([])
                                         for k, v in rows.items()})
        print(f"[d3] q{qi} 完成:{cfg.d3_groups_per_question} 组(退化 {n_deg})")
    del student
    gc.collect()
    torch.cuda.empty_cache()
