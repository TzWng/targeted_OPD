# -*- coding: utf-8 -*-
"""E1/E2 探针:跨题池化陪审团 + 蒸馏符号头(讨论 2026-07-16 剩余两提案)。

E1(池化,"免费扩大有效陪审团"):ι̂ 对陪审团 G 线性 ⇒ 每个陪审团对评测 token
的打分单独落盘,条件在分析期线性组合:
  C0 = 本题 8 条(≈ 训练现状,锚点应对上 d2 的 G=8);
  C1 = 本题 8 + 其余 7 题各 8(把整个训练 batch 池化,零额外采样);
  C2 = 本题 8 + 其余 7 题各 32(共享池加大,≈ EMA/复用旧 rollout 的稳态);
  C2m = C2 的每题等权(均值)口径;
  C3 = 只用其余 7 题(诊断:全局分量是否存在,≈50% ⇒ 符号信号是题内局部的)。
评测 token 与陪审团成员不相交(每题固定 32 条评测 rollout),天然无自项、免留一。

E2(蒸馏符号头,"摊销大陪审团"):缓存评测+额外 rollout 的末层隐状态 h_t,
训练轻头预测 R=256 参考符号,留一题交叉验证(8 折)。模型:
  (a) 加权逻辑回归 [h,d,logp];(b) MLP [h,emb(y),d,logp];(c) MLP 去 h(捷径检查)。
判读:留出题质量加权准确率 ≥ G=32 档 ⇒ 摊销可行;仅 ≥ G=8 档 ⇒ 与池化组合;≈50% ⇒ 否决。

用法:python -m probe.e_pool_head --config config.yaml --out <dir> --stage all
阶段:cache(GPU)→ e1(GPU)→ e2(GPU 小训练)→ report(CPU)。有产物即跳过。
"""
import argparse
import json
import os

import numpy as np
import torch

from .analysis import _join, _pct

N_EVAL, TOK_EVAL = 32, 128       # E1 评测:每题 rollout 数 × 每条采样 token 数
N_HEAD, TOK_HEAD = 32, 256       # E2 额外 h-only 池(与评测池合并训练)
OWN_DRAWS, OTH_DRAWS = 16, 4     # 本题陪审团重复数;别题陪审团抽取数
OTH_SIZES = (8, 32)
CHUNK = 1024                     # 打分时的 token 分块


def _qids(cfg):
    return [m["idx"] for m in json.load(open(os.path.join(cfg.out_dir, "questions.json")))]


# ------------------------------------------------------------ 纯数组工具
def pick_indices(R, rng):
    """评测 / 头额外 / 陪审团池 三者互不相交。"""
    perm = rng.permutation(R)
    return (np.sort(perm[:N_EVAL]), np.sort(perm[N_EVAL:N_EVAL + N_HEAD]),
            np.sort(perm[N_EVAL + N_HEAD:]))


def draw_jury(rng, pool, size, verdicts, max_tries=50):
    """从 pool 抽 size 条,组内中心化优势;退化(全对/全错)则重抽。"""
    for _ in range(max_tries):
        members = rng.choice(pool, size=size, replace=False)
        v = verdicts[members]
        if v.min() != v.max():
            adv_full = np.zeros(len(verdicts), np.float32)
            adv_full[members] = v - v.mean()
            return members.tolist(), adv_full
    return None, None


def subsample_positions(T, cap, rng):
    return np.sort(rng.choice(T, size=cap, replace=False)) if T > cap else np.arange(T)


@torch.no_grad()
def score_cached(G, h16, pi16, y, device):
    """ι̂ = (Gh)_y − ⟨π, Gh⟩,对缓存 token 分块计算。G[V,hd] 在 device 上。"""
    out = np.empty(len(y), np.float32)
    for s in range(0, len(y), CHUNK):
        h = torch.from_numpy(h16[s:s + CHUNK]).to(device).float()
        pi = torch.from_numpy(pi16[s:s + CHUNK]).to(device).float()
        yy = torch.from_numpy(y[s:s + CHUNK].astype(np.int64)).to(device)
        u = h @ G.t()
        out[s:s + CHUNK] = (u.gather(1, yy.unsqueeze(1)).squeeze(1)
                            - (pi * u).sum(-1)).cpu().numpy()
    return out


def combine_conditions(own, oth8, oth32):
    """own[K,n]、oth8/oth32[S,D,n] → 各条件 [K,n](第 k 次重复配第 k%D 组别题)。"""
    K, n = own.shape
    D = oth8.shape[1]
    j = np.arange(K) % D
    sum8 = oth8.sum(0)                     # [D,n]
    sum32 = oth32.sum(0)
    return {"C0": own,
            "C1": own + sum8[j],
            "C2": own + sum32[j],
            "C2m": own + (8.0 / 32.0) * sum32[j],
            "C3": sum32[j]}


def masswise_metrics(iota_hat_rows, s_ref, w):
    """质量加权符号准确率 + 实现增益率(对行取平均)。"""
    acc_n = acc_d = real_n = 0.0
    for row in np.atleast_2d(iota_hat_rows):
        s_hat = np.sign(row)
        m = (s_hat != 0) & (s_ref != 0)
        acc_n += float((w[m] * (s_hat[m] == s_ref[m])).sum())
        acc_d += float(w[m].sum())
        real_n += float((s_hat * s_ref * w).sum())     # sign·|dι| = sign·dι 的符号部分
    rows = len(np.atleast_2d(iota_hat_rows))
    return (acc_n / acc_d if acc_d else float("nan"),
            real_n / (rows * float(w.sum())) if w.sum() else float("nan"))


# ------------------------------------------------------------ 阶段 1:缓存
def run_cache_stage(cfg, device="cuda"):
    from .rollout import load_question
    from .stages import load_student, rollout_tensors
    done = all(os.path.exists(os.path.join(cfg.out_dir, f"q{qi}", "epool_cache.npz"))
               for qi in _qids(cfg))
    if done:
        print("[e-cache] 已全部存在,跳过")
        return
    tok, student = load_student(cfg, device)
    pad = tok.pad_token_id
    for qi in _qids(cfg):
        path = os.path.join(cfg.out_dir, f"q{qi}", "epool_cache.npz")
        if os.path.exists(path):
            continue
        meta, rolls = load_question(cfg, qi)
        rts = rollout_tensors(meta, rolls)
        rng = np.random.default_rng(cfg.seed * 3000 + qi)
        eval_idx, head_idx, _ = pick_indices(len(rts), rng)
        store = {k: [] for k in ("eval_h", "eval_pi", "eval_y", "eval_rollout", "eval_pos",
                                 "head_h", "head_y", "head_rollout", "head_pos")}
        for tag, idxs, cap, with_pi in (("eval", eval_idx, TOK_EVAL, True),
                                        ("head", head_idx, TOK_HEAD, False)):
            sel = [rts[i] for i in idxs]
            from .influence import iter_forward
            for local, probs, h, y, _ in iter_forward(student, pad, sel, device,
                                                      cfg.forward_batch):
                if probs is None:
                    continue
                pos = subsample_positions(int(y.shape[0]), cap,
                                          np.random.default_rng(cfg.seed * 7000 + qi * 97 + local))
                p = torch.from_numpy(pos).to(device)
                store[f"{tag}_h"].append(h[p].half().cpu().numpy())
                store[f"{tag}_y"].append(y[p].cpu().numpy().astype(np.int32))
                store[f"{tag}_rollout"].append(np.full(len(pos), idxs[local], np.int32))
                store[f"{tag}_pos"].append(pos.astype(np.int32))
                if with_pi:
                    store["eval_pi"].append(probs[p].half().cpu().numpy())
                del probs, h
        np.savez(path, **{k: np.concatenate(v) for k, v in store.items()},
                 eval_idx=eval_idx, head_idx=head_idx)
        print(f"[e-cache] q{qi}:eval {sum(len(x) for x in store['eval_pos'])} tok,"
              f" head {sum(len(x) for x in store['head_pos'])} tok")
    del student
    torch.cuda.empty_cache()


# ------------------------------------------------------------ 阶段 2:E1 打分
def run_e1_stage(cfg, device="cuda"):
    from .influence import build_jury_G
    from .rollout import load_question
    from .stages import load_student, rollout_tensors
    qids = _qids(cfg)
    out_paths = {qi: os.path.join(cfg.out_dir, f"q{qi}", "epool_scores.npz") for qi in qids}
    if all(os.path.exists(p) for p in out_paths.values()):
        print("[e1] 已全部存在,跳过")
        return
    caches = {qi: dict(np.load(os.path.join(cfg.out_dir, f"q{qi}", "epool_cache.npz")))
              for qi in qids}
    tok, student = load_student(cfg, device)
    V, hd, pad = student.config.vocab_size, student.config.hidden_size, tok.pad_token_id
    n_tok = {qi: len(caches[qi]["eval_y"]) for qi in qids}
    own = {qi: np.zeros((OWN_DRAWS, n_tok[qi]), np.float32) for qi in qids}
    oth = {sz: {qi: np.zeros((len(qids) - 1, OTH_DRAWS, n_tok[qi]), np.float32)
                for qi in qids} for sz in OTH_SIZES}
    src_order = {qi: [q for q in qids if q != qi] for qi in qids}

    for src_q in qids:
        meta, rolls = load_question(cfg, src_q)
        rts = rollout_tensors(meta, rolls)
        verdicts = np.array([r["verdict"] for r in rolls], np.float32)
        rng = np.random.default_rng(cfg.seed * 5000 + src_q)
        pool = np.setdiff1d(np.arange(len(rts)),
                            np.concatenate([caches[src_q]["eval_idx"],
                                            caches[src_q]["head_idx"]]))
        jobs = ([("own", 8, k) for k in range(OWN_DRAWS)]
                + [("oth", sz, j) for sz in OTH_SIZES for j in range(OTH_DRAWS)])
        for kind, size, k in jobs:
            members, adv_full = draw_jury(rng, pool, size, verdicts)
            if members is None:
                print(f"[e1] ⚠ q{src_q} {kind}{size}#{k} 退化重抽超限,置零跳过")
                continue
            G = build_jury_G(student, pad, rts, adv_full, members, device, V, hd,
                             cfg.forward_batch)
            targets = [src_q] if kind == "own" else [q for q in qids if q != src_q]
            for tq in targets:
                c = caches[tq]
                sc = score_cached(G, c["eval_h"], c["eval_pi"], c["eval_y"], device)
                if kind == "own":
                    own[tq][k] = sc
                else:
                    oth[size][tq][src_order[tq].index(src_q), k] = sc
            del G
            torch.cuda.empty_cache()
        print(f"[e1] 源题 q{src_q} 完成({len(jobs)} 个陪审团)")
    for qi in qids:
        np.savez_compressed(out_paths[qi], own=own[qi], oth8=oth[8][qi],
                            oth32=oth[32][qi],
                            src_order=np.array(src_order[qi], np.int32))
    del student
    torch.cuda.empty_cache()
    print("[e1] 全部落盘")


# ------------------------------------------------------------ 阶段 3:E2 训练
def _head_dataset(cfg, qids):
    """合并 eval+head 两个池的 h,并 join tokens.npz 取 d/logp/参考符号/质量。"""
    data = {}
    for qi in qids:
        c = dict(np.load(os.path.join(cfg.out_dir, f"q{qi}", "epool_cache.npz")))
        ref = dict(np.load(os.path.join(cfg.out_dir, f"q{qi}", "tokens.npz")))
        parts = []
        for tag in ("eval", "head"):
            rows = {"rollout": c[f"{tag}_rollout"], "pos": c[f"{tag}_pos"]}
            idx = _join(ref, rows)
            parts.append(dict(h=c[f"{tag}_h"], y=c[f"{tag}_y"],
                              d=ref["d"][idx].astype(np.float32),
                              logp=ref["logp_s"][idx].astype(np.float32),
                              s=np.sign(ref["ref_iota"][idx]).astype(np.float32),
                              w=np.abs(ref["d"][idx] * ref["ref_iota"][idx]).astype(np.float32)))
        data[qi] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        keep = data[qi]["s"] != 0
        data[qi] = {k: v[keep] for k, v in data[qi].items()}
    return data


def _train_head(model, Xtr, ytr, wtr, device, epochs=15, lr=3e-4, bs=8192):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(ytr)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(*[x[b] for x in Xtr]).squeeze(-1), ytr[b], weight=wtr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


class _MLP(torch.nn.Module):
    def __init__(self, hd, V, use_h=True, emb_dim=32, hidden=256):
        super().__init__()
        self.use_h = use_h
        self.emb = torch.nn.Embedding(V, emb_dim)
        in_dim = (hd if use_h else 0) + emb_dim + 2
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1))

    def forward(self, h, y, sc):
        parts = ([h] if self.use_h else []) + [self.emb(y), sc]
        return self.net(torch.cat(parts, -1))


class _Logistic(torch.nn.Module):
    def __init__(self, hd):
        super().__init__()
        self.lin = torch.nn.Linear(hd + 2, 1)

    def forward(self, h, y, sc):
        return self.lin(torch.cat([h, sc], -1))


def run_e2_stage(cfg, device="cuda"):
    out_path = os.path.join(cfg.out_dir, "results", "e2_folds.json")
    if os.path.exists(out_path):
        print("[e2] 已存在,跳过")
        return
    qids = _qids(cfg)
    data = _head_dataset(cfg, qids)
    V = int(max(int(d["y"].max()) for d in data.values())) + 1
    hd = data[qids[0]]["h"].shape[1]
    torch.manual_seed(cfg.seed)
    folds = {}
    for held in qids:
        tr = [q for q in qids if q != held]
        cat = {k: np.concatenate([data[q][k] for q in tr]) for k in data[held]}
        Xtr = (torch.from_numpy(cat["h"]).float().to(device),
               torch.from_numpy(cat["y"].astype(np.int64)).to(device),
               torch.from_numpy(np.stack([cat["d"], cat["logp"]], -1)).to(device))
        ytr = torch.from_numpy((cat["s"] > 0).astype(np.float32)).to(device)
        wtr = torch.from_numpy(cat["w"] / cat["w"].mean()).to(device)
        te = data[held]
        Xte = (torch.from_numpy(te["h"]).float().to(device),
               torch.from_numpy(te["y"].astype(np.int64)).to(device),
               torch.from_numpy(np.stack([te["d"], te["logp"]], -1)).to(device))
        res = {}
        for name, model, lr in (("logistic", _Logistic(hd), 1e-2),
                                ("mlp", _MLP(hd, V), 3e-4),
                                ("mlp_no_h", _MLP(hd, V, use_h=False), 3e-4)):
            model = _train_head(model.to(device), Xtr, ytr, wtr, device, lr=lr)
            with torch.no_grad():
                s_hat = np.sign(model(*Xte).squeeze(-1).cpu().numpy() )
                s_tr = np.sign(model(*Xtr).squeeze(-1).cpu().numpy())
            acc, real = masswise_metrics(np.where(s_hat == 0, 1, s_hat), te["s"], te["w"])
            acc_in, _ = masswise_metrics(np.where(s_tr == 0, 1, s_tr), cat["s"], cat["w"])
            res[name] = {"acc": acc, "realized": real, "acc_train": acc_in,
                         "n_test": int(len(te["s"]))}
            del model
        folds[str(held)] = res
        print(f"[e2] 留出 q{held} 完成:", {k: f"{v['acc']:.3f}" for k, v in res.items()})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(folds, open(out_path, "w"), indent=1)


# ------------------------------------------------------------ 阶段 4:汇总
def run_e_report(cfg):
    qids = _qids(cfg)
    summary = {"conditions": {}, "n_tokens": 0}
    agg = {}
    opd_n = opd_d = 0.0
    for qi in qids:
        c = dict(np.load(os.path.join(cfg.out_dir, f"q{qi}", "epool_cache.npz")))
        sc = dict(np.load(os.path.join(cfg.out_dir, f"q{qi}", "epool_scores.npz")))
        ref = dict(np.load(os.path.join(cfg.out_dir, f"q{qi}", "tokens.npz")))
        idx = _join(ref, {"rollout": c["eval_rollout"], "pos": c["eval_pos"]})
        s_ref = np.sign(ref["ref_iota"][idx])
        d = ref["d"][idx].astype(np.float64)
        w = np.abs(d * ref["ref_iota"][idx])
        opd_n += float((d * ref["ref_iota"][idx]).sum())
        opd_d += float(w.sum())
        summary["n_tokens"] += len(s_ref)
        for name, rows in combine_conditions(sc["own"], sc["oth8"], sc["oth32"]).items():
            e = agg.setdefault(name, [0.0, 0.0, 0.0, 0])   # acc_n, acc_d, real_n, rows*wsum
            for row in rows:
                s_hat = np.sign(row)
                m = (s_hat != 0) & (s_ref != 0)
                e[0] += float((w[m] * (s_hat[m] == s_ref[m])).sum())
                e[1] += float(w[m].sum())
                e[2] += float((s_hat * s_ref * w).sum())
            e[3] += rows.shape[0] * float(w.sum())
    for name, (an, ad, rn, rd) in agg.items():
        summary["conditions"][name] = {"acc": an / ad if ad else float("nan"),
                                       "realized": rn / rd if rd else float("nan")}
    summary["opd_alignment"] = opd_n / opd_d if opd_d else float("nan")
    e2_path = os.path.join(cfg.out_dir, "results", "e2_folds.json")
    if os.path.exists(e2_path):
        folds = json.load(open(e2_path))
        summary["e2"] = {"folds": folds}
        for m in ("logistic", "mlp", "mlp_no_h"):
            accs = [folds[q][m]["acc"] for q in folds]
            reals = [folds[q][m]["realized"] for q in folds]
            ins = [folds[q][m]["acc_train"] for q in folds]
            summary["e2"][m] = {"acc_heldout_mean": float(np.mean(accs)),
                                "acc_heldout_min": float(np.min(accs)),
                                "realized_mean": float(np.mean(reals)),
                                "acc_train_mean": float(np.mean(ins))}
    res_dir = os.path.join(cfg.out_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    json.dump(summary, open(os.path.join(res_dir, "e_summary.json"), "w"),
              ensure_ascii=False, indent=1)
    _write_e_report(summary, os.path.join(res_dir, "e_report.md"))
    print(f"[e-report] 完成 → {res_dir}/e_report.md")
    return summary


COND_LABEL = {"C0": "本题 8(训练现状)", "C1": "本题 8 + 别题 7×8(批内池化,零成本)",
              "C2": "本题 8 + 别题 7×32(大共享池 ≈ EMA 稳态)",
              "C2m": "同 C2,每题等权(均值口径)", "C3": "只用别题 7×32(纯迁移诊断)"}


def _write_e_report(s, path):
    L = ["# E1/E2 探针 —— 池化陪审团 & 蒸馏符号头\n",
         f"评测 token 数 {s['n_tokens']:,}(每题固定评测 rollout,与陪审团不相交,免留一);"
         f"OPD 基线对齐率 {s['opd_alignment'] * 100:+.1f}%。\n",
         "## E1 —— 跨题池化(质量加权)\n",
         "| 条件 | 符号准确率 | 实现增益率 |", "|---|---|---|"]
    for name in ("C0", "C1", "C2", "C2m", "C3"):
        c = s["conditions"].get(name)
        if c:
            L.append(f"| {COND_LABEL[name]} | **{_pct(c['acc'])}** | {c['realized'] * 100:+.1f}% |")
    c0 = s["conditions"].get("C0", {}).get("acc", float("nan"))
    c1 = s["conditions"].get("C1", {}).get("acc", float("nan"))
    c3 = s["conditions"].get("C3", {}).get("acc", float("nan"))
    L.append("\n**判读(预写):**\n")
    if c1 == c1 and c0 == c0:
        gain = (c1 - c0) * 100
        if gain >= 3:
            L.append(f"- C1−C0 = +{gain:.1f}pp ⇒ **批内池化免费有效**,训练侧应立即把 ĝ_Q 从组内改成全批。")
        elif gain > 0.5:
            L.append(f"- C1−C0 = +{gain:.1f}pp:有小提升,免费顺手改,但不解决主问题。")
        else:
            L.append(f"- C1−C0 = {gain:+.1f}pp ⇒ 批内池化无效:别题方向对本题 token 无增益。")
    if c3 == c3:
        L.append(f"- C3(纯别题)= {_pct(c3)}:" +
                 ("> 55% ⇒ 存在跨题全局分量,池化/EMA 路线有物理基础。" if c3 > 0.55 else
                  "≈ 50% ⇒ 符号信号基本是题内局部的,池化只能靠方差平均起作用。"))
    if "e2" in s:
        L.append("\n## E2 —— 蒸馏符号头(留一题交叉验证,8 折)\n")
        L.append("| 模型 | 留出题准确率(均值) | 最差折 | 实现增益率 | 训练集准确率 |")
        L.append("|---|---|---|---|---|")
        for m, lab in (("logistic", "逻辑回归 [h,d,logp]"), ("mlp", "MLP [h,emb(y),d,logp]"),
                       ("mlp_no_h", "MLP 去 h(捷径检查)")):
            e = s["e2"][m]
            L.append(f"| {lab} | **{_pct(e['acc_heldout_mean'])}** | {_pct(e['acc_heldout_min'])} | "
                     f"{e['realized_mean'] * 100:+.1f}% | {_pct(e['acc_train_mean'])} |")
        best = max(("logistic", "mlp"), key=lambda m: s["e2"][m]["acc_heldout_mean"])
        acc = s["e2"][best]["acc_heldout_mean"]
        L.append("\n**判读(预写):** 对照 d2:G=8≈58.7/64.5%,G=32≈68/74%,G=64≈76/81%。\n")
        if acc >= 0.68:
            L.append(f"- 留出题 {_pct(acc)} ≥ G=32 档 ⇒ **摊销可行**:隔 K 步大陪审团打标 + 轻头,每步零成本拿大陪审团精度。")
        elif acc >= 0.60:
            L.append(f"- 留出题 {_pct(acc)} 介于 G=8 与 G=32 之间 ⇒ 有信号,建议与池化组合或加特征(教师 logits)再试。")
        else:
            L.append(f"- 留出题 {_pct(acc)} ≈ G=8 或更低 ⇒ h 中无足够可迁移信号,摊销路线否决。")
        no_h = s["e2"]["mlp_no_h"]["acc_heldout_mean"]
        if no_h == no_h and no_h > acc - 0.02:
            L.append(f"- ⚠ 去 h 的 MLP 达 {_pct(no_h)},与含 h 相当:头可能在学 (y,d) 捷径而非隐状态信号,解释时注意。")
    L.append("\n*由 probe/e_pool_head.py 自动生成;原始数据 q*/epool_cache.npz、epool_scores.npz。*")
    with open(path, "w") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stage", default="all", choices=["cache", "e1", "e2", "report", "all"])
    args = ap.parse_args()
    from .config import load_config
    cfg = load_config(args.config, out_dir=args.out)
    stages = [args.stage] if args.stage != "all" else ["cache", "e1", "e2", "report"]
    for st in stages:
        print(f"\n===== 阶段: {st} =====")
        if st == "cache":
            run_cache_stage(cfg)
        elif st == "e1":
            run_e1_stage(cfg)
        elif st == "e2":
            run_e2_stage(cfg)
        elif st == "report":
            run_e_report(cfg)


if __name__ == "__main__":
    main()
