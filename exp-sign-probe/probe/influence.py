# -*- coding: utf-8 -*-
"""影响力核心数学(被审计对象),与论文式 (15)/算法 1 及训练实现一致:

  G      = Σ_i a_i Σ_t (e_{y_t} − π_t) h_tᵀ                ∈ R^{V×h}(陪审团)
  ι̂(t)   = (G h_t)_{y_t} − ⟨π_t, G h_t⟩                     (打分)
  留一    = ι̂ − a_j · corr_j(t),corr 用同一 rollout 的 Gram 恒等式

全部为纯数组函数(fp32),tests/test_influence_small.py 用暴力实现逐项对照。
模型侧只有 iter_forward / teacher_logp 两个包装。"""
import torch
import torch.nn.functional as F


# --------------------------------------------------------------- 纯数组核心
def accumulate_G(G, probs_c, h_c, y_c, a_c):
    """G[V,h] += Σ_c a_c (e_{y_c} − π_c) h_cᵀ(就地,chunk 级)。"""
    G -= (a_c.unsqueeze(1) * probs_c).t() @ h_c
    G.index_add_(0, y_c, a_c.unsqueeze(1) * h_c)


def iota_from_G(G, probs_c, h_c, y_c):
    """ι̂ = (G h)_y − ⟨π, G h⟩,对一个 chunk 的 token 同时计算。→ [c]"""
    u = h_c @ G.t()                                   # [c, V]
    return u.gather(1, y_c.unsqueeze(1)).squeeze(1) - (probs_c * u).sum(-1)


def own_correction(probs, h, y):
    """rollout 自身的 Gram 修正(未乘优势):
    corr[t] = (C h_t)_{y_t} − ⟨π_t, C h_t⟩,C = Σ_s (e_{y_s} − π_s) h_sᵀ。
    probs [T,V] fp32、h [T,hd] fp32、y [T]。→ [T]
    留一:ι̂^{(−j)}(t) = iota_from_G(G_full)(t) − a_j · corr[t](t ∈ rollout j)。"""
    C = h @ h.t()                                     # [T,T] ⟨h_s,h_t⟩
    M1 = probs[:, y]                                  # M1[s,t] = π_s(y_t)
    PP = probs @ probs.t()                            # ⟨π_s,π_t⟩
    EQ = (y.unsqueeze(1) == y.unsqueeze(0)).float()
    return (C * (EQ - M1 - M1.t() + PP)).sum(0)


# ------------------------------------------------------------- 暴力对照实现
def brute_G(probs, h, y, a):
    V, hd = probs.shape[1], h.shape[1]
    G = torch.zeros(V, hd, dtype=torch.float32)
    for i in range(len(y)):
        e = torch.zeros(V)
        e[y[i]] = 1.0
        G += a[i] * torch.outer(e - probs[i], h[i])
    return G


def brute_iota(G, probs_t, h_t, y_t):
    u = G @ h_t
    return float(u[int(y_t)] - probs_t @ u)


# ----------------------------------------------------------------- 模型包装
@torch.no_grad()
def iter_forward(model, pad_id, rollouts, device, batch=4, chunk_logits=True):
    """逐 rollout 产出 (idx, probs[T,V] fp32, h[T,hd] fp32, y[T], logp_s[T])。
    rollouts: [{"ids": LongTensor[L], "plen": int}],内部按 batch 右填充成批前向。
    位置约定与训练一致:位置 j 预测 ids[j+1];completion = ids[plen:]。"""
    model.eval()
    for s in range(0, len(rollouts), batch):
        group = rollouts[s:s + batch]
        L = max(g["ids"].shape[0] for g in group)
        ids = torch.full((len(group), L), pad_id, dtype=torch.long, device=device)
        for j, g in enumerate(group):
            ids[j, :g["ids"].shape[0]] = g["ids"].to(device)
        attn = (ids != pad_id).long()
        # pad_id 可能与真实 token 相同(右填充时安全:真实段在左侧连续)
        for j, g in enumerate(group):
            attn[j, :g["ids"].shape[0]] = 1
        out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
        logits = out.logits                            # [B, L, V]
        hidden = out.hidden_states[-1]                 # [B, L, hd] 末层 norm 后 = 论文的 h_t
        for j, g in enumerate(group):
            Lj, plen = g["ids"].shape[0], g["plen"]
            T = Lj - plen
            if T <= 0:
                yield s + j, None, None, None, None
                continue
            pos = torch.arange(plen - 1, Lj - 1, device=device)
            lg = logits[j, pos].float()                # [T, V]
            probs = F.softmax(lg, dim=-1)
            y = g["ids"][plen:].to(device)
            logp = lg.log_softmax(-1).gather(1, y.unsqueeze(1)).squeeze(1)
            h = hidden[j, pos].float()
            yield s + j, probs, h, y, logp
        del out, logits, hidden


@torch.no_grad()
def teacher_logp(model, pad_id, rollouts, device, batch=4):
    """教师在采到 token 上的 logP(逐行 fp32 log_softmax),→ 每 rollout 一个 [T]。"""
    model.eval()
    results = [None] * len(rollouts)
    for s in range(0, len(rollouts), batch):
        group = rollouts[s:s + batch]
        L = max(g["ids"].shape[0] for g in group)
        ids = torch.full((len(group), L), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(group), L), dtype=torch.long, device=device)
        for j, g in enumerate(group):
            ids[j, :g["ids"].shape[0]] = g["ids"].to(device)
            attn[j, :g["ids"].shape[0]] = 1
        logits = model(input_ids=ids, attention_mask=attn).logits
        for j, g in enumerate(group):
            Lj, plen = g["ids"].shape[0], g["plen"]
            pos = torch.arange(plen - 1, Lj - 1, device=device)
            lg = logits[j, pos].float().log_softmax(-1)
            y = g["ids"][plen:].to(device)
            results[s + j] = lg.gather(1, y.unsqueeze(1)).squeeze(1).cpu()
        del logits
    return results


# ------------------------------------------------------------- 陪审团 G 构建
@torch.no_grad()
def build_jury_G(model, pad_id, rollouts, adv, member_idx, device, V, hd,
                 batch=4, halves=False):
    """对 member_idx 里的 rollout 前向并累加 G(优势 a 按传入的 adv,训练约定)。
    halves=True 时同时返回 (G, G_half0, G_half1),半批按成员在 member_idx
    中的位置奇偶划分(对应训练代码 (bi % G) % 2 的组内奇偶)。"""
    G = torch.zeros(V, hd, dtype=torch.float32, device=device)
    Gh = [torch.zeros(V, hd, dtype=torch.float32, device=device),
          torch.zeros(V, hd, dtype=torch.float32, device=device)] if halves else None
    members = [rollouts[i] for i in member_idx]
    half_of = {ri: (k % 2) for k, ri in enumerate(member_idx)}
    for local, probs, h, y, _ in iter_forward(model, pad_id, members, device, batch):
        if probs is None:
            continue
        ri = member_idx[local]
        a = torch.full((y.shape[0],), float(adv[ri]), device=device)
        accumulate_G(G, probs, h, y, a)
        if halves:
            accumulate_G(Gh[half_of[ri]], probs, h, y, a)
        del probs, h
    return (G, Gh[0], Gh[1]) if halves else G
