# -*- coding: utf-8 -*-
"""阶段 5:汇总统计 → results/summary.json + results/report.md(中文自动判读)。
口径:
  * 符号准确率一律与参考真值(R=256 大陪审团、精确留一)比较;
  * 三种权重:unweighted / |ι_ref| / |d·ι_ref|(论文的"质量",判读用后者);
  * 只统计双方符号皆非零的 token;退化陪审团单独计数。"""
import json
import os

import numpy as np

KEY_MULT = 1 << 14   # (rollout, pos) → 单键;pos < 16384


def _key(rollout, pos):
    return rollout.astype(np.int64) * KEY_MULT + pos.astype(np.int64)


def _join(ref, rows):
    """按 (rollout,pos) 把 rows 对齐到参考表,返回参考行索引。"""
    rk = _key(ref["rollout"], ref["pos"])
    order = np.argsort(rk)
    rk_sorted = rk[order]
    qk = _key(rows["rollout"], rows["pos"])
    idx = np.searchsorted(rk_sorted, qk)
    idx = np.clip(idx, 0, len(rk_sorted) - 1)
    ok = rk_sorted[idx] == qk
    if not ok.all():
        raise RuntimeError(f"join 失败:{(~ok).sum()} 个 token 在参考表中缺失")
    return order[idx]


def _acc(s_hat, s_ref, w):
    m = (s_hat != 0) & (s_ref != 0)
    if m.sum() == 0:
        return float("nan"), 0
    wm = w[m]
    return float((wm * (s_hat[m] == s_ref[m])).sum() / wm.sum()), int(m.sum())


def _agree(sa, sb, w):
    m = (sa != 0) & (sb != 0)
    if m.sum() == 0:
        return float("nan"), 0
    wm = w[m]
    return float((wm * (sa[m] == sb[m])).sum() / wm.sum()), int(m.sum())


def _weights(ref, idx):
    return {
        "unweighted": np.ones(len(idx), np.float64),
        "w_iota": np.abs(ref["ref_iota"][idx]),
        "w_mass": np.abs(ref["d"][idx] * ref["ref_iota"][idx]),
    }


def run_analysis(cfg):
    qids = [m["idx"] for m in json.load(open(os.path.join(cfg.out_dir, "questions.json")))]
    res_dir = os.path.join(cfg.out_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    refs = {qi: dict(np.load(os.path.join(cfg.out_dir, f"q{qi}", "tokens.npz")))
            for qi in qids}

    summary = {"config_note": f"dataset={cfg.dataset} R={cfg.rollouts_per_question} "
                              f"cap={cfg.max_new_tokens} adapter={cfg.adapter_path}",
               "n_questions": len(qids)}

    # ---------- 参考自稳定性(128 vs 128) ----------
    st = {k: [0.0, 0.0] for k in ("unweighted", "w_iota", "w_mass")}
    for qi in qids:
        r = refs[qi]
        idx = np.arange(len(r["pos"]))
        W = _weights(r, idx)
        for k, w in W.items():
            m = (r["signA"] != 0) & (r["signB"] != 0)
            st[k][0] += float((w[m] * (r["signA"][m] == r["signB"][m])).sum())
            st[k][1] += float(w[m].sum())
    summary["reference_stability"] = {k: v[0] / v[1] for k, v in st.items()}
    trunc_rate = float(np.mean(np.concatenate([refs[qi]["trunc"] for qi in qids])))
    summary["token_trunc_rate"] = trunc_rate

    # ---------- D2:按陪审团大小 ----------
    d2 = {}
    deg = {}
    for qi in qids:
        for j in json.load(open(os.path.join(cfg.out_dir, f"q{qi}", "d2_juries.json"))):
            deg.setdefault(j["size"], []).append(j["degenerate"])
    for size in cfg.jury_sizes:
        acc = {k: [0.0, 0.0] for k in ("unweighted", "w_iota", "w_mass")}
        acc_gated = {k: [0.0, 0.0] for k in ("unweighted", "w_iota", "w_mass")}
        n_valid, n_gated = 0, 0
        decile_hits = np.zeros(10)
        decile_wsum = np.zeros(10)
        ttype_acc = {t: [0.0, 0.0] for t in range(4)}
        for qi in qids:
            rows = dict(np.load(os.path.join(cfg.out_dir, f"q{qi}", "d2.npz")))
            if rows["size"].size == 0:
                continue
            sel = rows["size"] == size
            if sel.sum() == 0:
                continue
            sub = {k: v[sel] for k, v in rows.items()}
            idx = _join(refs[qi], sub)
            s_ref = np.sign(refs[qi]["ref_iota"][idx])
            s_hat = np.sign(sub["iota_hat"])
            W = _weights(refs[qi], idx)
            valid = (s_hat != 0) & (s_ref != 0)
            n_valid += int(valid.sum())
            for k, w in W.items():
                acc[k][0] += float((w[valid] * (s_hat[valid] == s_ref[valid])).sum())
                acc[k][1] += float(w[valid].sum())
            gated = valid & (sub["alt_sign"] != 0) & (s_hat == sub["alt_sign"])
            n_gated += int(gated.sum())
            for k, w in W.items():
                acc_gated[k][0] += float((w[gated] * (s_hat[gated] == s_ref[gated])).sum())
                acc_gated[k][1] += float(w[gated].sum())
            # |ι̂| 十分位校准(质量加权命中)
            mag = np.abs(sub["iota_hat"][valid])
            if mag.size:
                qcuts = np.quantile(mag, np.linspace(0, 1, 11))
                bins = np.clip(np.searchsorted(qcuts[1:-1], mag), 0, 9)
                hit = (s_hat[valid] == s_ref[valid]) * W["w_mass"][valid]
                for b in range(10):
                    decile_hits[b] += hit[bins == b].sum()
                    decile_wsum[b] += W["w_mass"][valid][bins == b].sum()
            tt = refs[qi]["ttype"][idx]
            for t in range(4):
                mt = valid & (tt == t)
                ttype_acc[t][0] += float((W["w_mass"][mt] * (s_hat[mt] == s_ref[mt])).sum())
                ttype_acc[t][1] += float(W["w_mass"][mt].sum())
        d2[str(size)] = {
            "degenerate_jury_rate": float(np.mean(deg.get(size, [0]))),
            "n_tokens": n_valid,
            "acc": {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in acc.items()},
            "gate_coverage": (n_gated / n_valid if n_valid else float("nan")),
            "acc_gated": {k: (v[0] / v[1] if v[1] else float("nan"))
                          for k, v in acc_gated.items()},
            "decile_acc_w_mass": [float(decile_hits[b] / decile_wsum[b])
                                  if decile_wsum[b] else float("nan") for b in range(10)],
            "ttype_acc_w_mass": {["digit", "symbol", "word", "other"][t]:
                                 (v[0] / v[1] if v[1] else float("nan"))
                                 for t, v in ttype_acc.items()},
        }
    summary["d2"] = d2

    # ---------- D3:三票型 ----------
    agg = {k: [0.0, 0.0] for k in
           ("agree12", "agree32", "acc1", "acc2", "acc3", "acc_gate12", "cov_gate12")}
    n_tok = 0
    for qi in qids:
        rows = dict(np.load(os.path.join(cfg.out_dir, f"q{qi}", "d3.npz")))
        if rows["group"].size == 0:
            continue
        idx = _join(refs[qi], rows)
        s_ref = np.sign(refs[qi]["ref_iota"][idx])
        w = np.abs(refs[qi]["d"][idx] * refs[qi]["ref_iota"][idx])
        s1, s2, s3 = (np.sign(rows[k]) for k in ("v1", "v2", "v3"))
        n_tok += len(s1)
        for name, (a, b) in (("agree12", (s1, s2)), ("agree32", (s3, s2))):
            m = (a != 0) & (b != 0)
            agg[name][0] += float((w[m] * (a[m] == b[m])).sum())
            agg[name][1] += float(w[m].sum())
        for name, s in (("acc1", s1), ("acc2", s2), ("acc3", s3)):
            m = (s != 0) & (s_ref != 0)
            agg[name][0] += float((w[m] * (s[m] == s_ref[m])).sum())
            agg[name][1] += float(w[m].sum())
        gate = (s1 != 0) & (s2 != 0) & (s1 == s2) & (s_ref != 0)
        agg["acc_gate12"][0] += float((w[gate] * (s1[gate] == s_ref[gate])).sum())
        agg["acc_gate12"][1] += float(w[gate].sum())
        valid = (s1 != 0) & (s_ref != 0)
        agg["cov_gate12"][0] += float(gate.sum())
        agg["cov_gate12"][1] += float(valid.sum())
    d3 = {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in agg.items()}
    d3["inflation"] = d3["agree12"] - d3["agree32"]
    d3["n_tokens"] = n_tok
    summary["d3"] = d3

    with open(os.path.join(res_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    _write_report(cfg, summary, os.path.join(res_dir, "report.md"))
    print(f"[analyze] 完成 → {res_dir}/summary.json, report.md")
    return summary


# ------------------------------------------------------------------ 报告
def _pct(x):
    return "—" if (x != x) else f"{100 * x:.1f}%"


def _write_report(cfg, s, path):
    L = []
    L.append("# D2/D3 符号准确率探针 —— 结果报告\n")
    L.append(f"配置:{s['config_note']};探针题 {s['n_questions']} 道;"
             f"token 级截断率 {_pct(s['token_trunc_rate'])}(生成上限已放宽)。\n")
    rs = s["reference_stability"]
    L.append("## 0. 参考真值自稳定性(128 条 vs 128 条)\n")
    L.append(f"- 质量加权(|d·ι|):**{_pct(rs['w_mass'])}** · |ι| 加权 {_pct(rs['w_iota'])}"
             f" · 不加权 {_pct(rs['unweighted'])}")
    if rs["w_mass"] < 0.85:
        L.append("- ⚠ 参考自稳定性 < 85%:R=256 的参考本身还不够稳,下面的绝对数值要保守解读"
                 "(相对比较仍有效);可考虑加大 R。")
    else:
        L.append("- 参考足够稳定,可以作为真值使用。")
    L.append("\n## 1. D2 —— 符号准确率 vs 陪审团大小(质量加权,|d·ι_ref|)\n")
    L.append("| 陪审团 G | 退化率 | 符号准确率 | 门控后准确率 | 门控保留率 | token 数 |")
    L.append("|---|---|---|---|---|---|")
    for size in cfg.jury_sizes:
        d = s["d2"][str(size)]
        L.append(f"| {size} | {_pct(d['degenerate_jury_rate'])} | "
                 f"**{_pct(d['acc']['w_mass'])}** | {_pct(d['acc_gated']['w_mass'])} | "
                 f"{_pct(d['gate_coverage'])} | {d['n_tokens']:,} |")
    g8 = s["d2"].get("8", s["d2"][str(cfg.jury_sizes[0])])
    a8 = g8["acc"]["w_mass"]
    L.append("\n**判读(按诊断计划 D2 的预写标准):**\n")
    if a8 == a8 and a8 < 0.60:
        L.append(f"- G=8 质量加权准确率 {_pct(a8)} **< 60% ⇒ F1 实锤**:训练配置下的影响力符号"
                 "接近抛硬币。先修估计器(更大的陪审团、换投影、G 的滑动平均),再谈任何 targeted 战役。")
    elif a8 == a8 and a8 < 0.70:
        L.append(f"- G=8 质量加权准确率 {_pct(a8)},介于 60–70%:有信号但很弱。看 G=32/64 是否"
                 "明显抬升——若抬升,'定符号用大陪审团、训练只回传少数'是明确的解法。")
    elif a8 == a8:
        L.append(f"- G=8 质量加权准确率 {_pct(a8)} ≥ 70%:符号本身可用。结果不佳的原因更可能在"
                 "别处(沙盒推不动 F2、截断 F3、漂移 F6)——按计划回到阶段 C。")
    L.append("- 十分位校准(G=8,按 |ι̂| 从小到大,质量加权):"
             + " ".join(_pct(x) for x in g8["decile_acc_w_mass"]))
    L.append("- 按 token 类型(G=8):" + " · ".join(
        f"{k} {_pct(v)}" for k, v in g8["ttype_acc_w_mass"].items()))
    d3 = s["d3"]
    L.append("\n## 2. D3 —— 分半一致率的水分(G=8 训练组,质量加权)\n")
    L.append(f"| 量 | 值 | 含义 |")
    L.append("|---|---|---|")
    L.append(f"| agree12(训练口径:全组留一 vs 对面半批) | **{_pct(d3['agree12'])}** | "
             f"训练日志里 agree≈0.76–0.77 的对应物(校准点) |")
    L.append(f"| agree32(真独立:本半批去己 vs 对面半批) | **{_pct(d3['agree32'])}** | "
             f"去掉共享数据后的一致率 |")
    L.append(f"| 水分 = agree12 − agree32 | **{d3['inflation'] * 100:.1f}pp** | "
             f"共享陪审团带来的虚高 |")
    L.append(f"| acc1 / acc2 / acc3(各票 vs 参考真值) | {_pct(d3['acc1'])} / "
             f"{_pct(d3['acc2'])} / {_pct(d3['acc3'])} | 票的真实准确率 |")
    L.append(f"| 门控(1、2 票一致)后的准确率 · 保留率 | {_pct(d3['acc_gate12'])} · "
             f"{_pct(d3['cov_gate12'])} | 训练门控实际买到的提升 |")
    L.append("\n**判读:**agree12 若与训练日志的 ~0.76 接近,说明本探针对训练机制的复现是忠实的;"
             "acc1(≈门控前的真准确率)与 agree12 的差距 = 之前被『互相印证』掩盖的部分;"
             "门控后准确率相对 acc1 的提升幅度 × 保留率,决定门控是否值得保留。\n")
    L.append(f"\n*由 probe/analysis.py 自动生成;原始数据在 q*/tokens.npz、d2.npz、d3.npz。*")
    with open(path, "w") as f:
        f.write("\n".join(L))
