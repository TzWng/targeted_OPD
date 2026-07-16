# -*- coding: utf-8 -*-
"""块级(block-level)符号重分析 —— 纯离线,只读已有 q*/tokens.npz、q*/d2.npz。

背景(讨论 2026-07-16):D2 显示 G=8 逐 token 符号不可用、G=64 太贵。
提案:块内共享一个符号,信号按块长 T 线性累加、陪审团噪声只按 √T 增长,
信噪比提升 ~√T,而采样开销不变。本脚本用已落盘的探针数据回答三个问题:

  B0 真符号成片吗?—— 相邻符号一致率、游程长度、块级证书保留率 coherence(T)
     (= 最优块常值符号能保住的证书质量比例;若真符号椒盐状,分块把信号也平均掉);
  B1 块级符号准确率 / 实现增益率 vs (陪审团 G × 块方案);
  B2 训练档 G 下的置信度–覆盖率曲线(低置信块回退 0 或回退标准 OPD)。

口径:
  * 块值 v_B = Σ_{t∈B} |d_t|·ι_t。增益对齐:块常值符号奖励 r_t = sign(v_B)·|d_t|
    的一阶增益 = Σ_B sign·v_B,故最优块符号 = sign(v_B^ref),估计 = sign(v_B^hat);
  * 块权重 = |v_B^ref|(块级"质量";T=1 时严格退化为 analysis.py 的 |d·ι| 口径);
  * 实现增益率 = 一阶增益 / token 级证书上限 Σ_t|d·ι_ref| ∈ [−1,1]
    (T=1 全覆盖时 = 2·准确率 − 1);OPD 基线 = Σ_t d·ι_ref / Σ_t|d·ι_ref|;
  * 与 analysis.py 相同:只统计双方符号非零的块;权重一律用参考真值。
  * 注意:参考半批只存了符号没存数值,块级参考自稳定性算不出来——但块内求和
    对参考噪声同样有 √T 平均效应,块级参考只会比 token 级更稳,判读时记住即可。

用法(登录节点 CPU 即可,numpy 为主,几分钟):
  python -m probe.block_analysis --config config.yaml --out /scratch/$USER/opd_sign_probe/run1
输出:<out>/results/block_report.md + block_summary.json。
"""
import argparse
import json
import os

import numpy as np

from .analysis import _join, _pct

BIG_JURY = 1 << 42          # jury 键位移(jury < 2^5,bid < 2^30)
BIG_ROLL = 1 << 20          # rollout 键位移(pos < 2^20)

FIXED_T = [1, 2, 4, 8, 16, 32, 64]
COVERAGES = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
SCHEME_LABEL = {"nl": "换行分段", "roll": "整条 rollout"}
SCHEME_LABEL.update({f"T{t}": (f"T={t}" + ("(逐 token)" if t == 1 else ""))
                     for t in FIXED_T})


# ------------------------------------------------------------ 块 id 构造
def fixed_block_ids(rollout, pos, T):
    return rollout.astype(np.int64) * BIG_ROLL + pos.astype(np.int64) // T


def rollout_block_ids(rollout):
    return rollout.astype(np.int64) * BIG_ROLL


def newline_block_ids(rollout, pos, is_nl):
    """换行 token 结束一段;段 id 在题内全局唯一(不跨 rollout)。"""
    order = np.lexsort((pos, rollout))
    r_o = rollout[order]
    b_o = is_nl[order].astype(bool)
    start = np.ones(r_o.size, bool)
    start[1:] = (r_o[1:] != r_o[:-1]) | b_o[:-1]
    seg = np.empty(r_o.size, np.int64)
    seg[order] = np.cumsum(start) - 1
    return seg


def block_reduce(key, cols):
    """按块 id 分组求和;返回 (块数, {列名: 每块和})。"""
    uniq, inv = np.unique(key, return_inverse=True)
    return uniq.size, {k: np.bincount(inv, weights=v.astype(np.float64),
                                      minlength=uniq.size)
                       for k, v in cols.items()}


def build_scheme_ids(ref, is_nl):
    ids = {f"T{t}": fixed_block_ids(ref["rollout"], ref["pos"], t) for t in FIXED_T}
    if is_nl is not None:
        ids["nl"] = newline_block_ids(ref["rollout"], ref["pos"], is_nl)
    ids["roll"] = rollout_block_ids(ref["rollout"])
    return ids


# ------------------------------------------------------------ 累加器
def new_state():
    return {
        "agg": {},                 # (size, scheme) → 累加项
        "cov": {},                 # scheme → 训练档 G 的每块数组(算覆盖率曲线)
        "cov_den": 0.0,            # 训练档 G 的 token 证书总质量
        "coh_num": {}, "coh_den": 0.0,      # B0:coherence(方案)
        "opd_num": 0.0, "opd_den": 0.0,     # OPD 基线对齐率
        "adj_num_u": 0.0, "adj_den_u": 0.0,  # 相邻符号一致率(不加权)
        "adj_num_w": 0.0, "adj_den_w": 0.0,  # (质量加权)
        "run_lens": [],            # 各题游程长度数组
        "runlen_num_w": 0.0, "runlen_den_w": 0.0,  # 质量加权"所在游程长度"
        "nl_lens": [],             # 换行段长度数组
    }


def _entry(st, size, scheme):
    return st["agg"].setdefault((int(size), scheme), dict(
        acc_num=0.0, acc_den=0.0, pacc_num=0.0, pacc_den=0.0,
        real_num=0.0, tok_den=0.0, n_blocks=0))


def _accumulate_runs(ref, st):
    order = np.lexsort((ref["pos"], ref["rollout"]))
    s = np.sign(ref["ref_iota"]).astype(np.int8)[order]
    r = ref["rollout"][order]
    w = np.abs(ref["d"].astype(np.float64) * ref["ref_iota"])[order]
    same = r[1:] == r[:-1]
    both = same & (s[1:] != 0) & (s[:-1] != 0)
    eq = s[1:] == s[:-1]
    st["adj_num_u"] += float(eq[both].sum())
    st["adj_den_u"] += float(both.sum())
    ww = w[1:]
    st["adj_num_w"] += float((ww * eq)[both].sum())
    st["adj_den_w"] += float(ww[both].sum())
    start = np.ones(s.size, bool)
    start[1:] = (~same) | (s[1:] != s[:-1])
    rid = np.cumsum(start) - 1
    lens = np.bincount(rid)
    st["run_lens"].append(lens)
    st["runlen_num_w"] += float((w * lens[rid]).sum())
    st["runlen_den_w"] += float(w.sum())


def analyze_question(ref, d2, is_nl, train_g, st):
    """处理一道题:B0 全 256 条参考 + B1/B2 各陪审团。纯 numpy,可单测。"""
    scheme_ids = build_scheme_ids(ref, is_nl)
    d64 = ref["d"].astype(np.float64)
    i64 = ref["ref_iota"].astype(np.float64)
    vref_tok = np.abs(d64) * i64
    w_tok = np.abs(d64 * i64)

    # ---- B0:coherence + 游程 + OPD 基线(全部 256 条参考 rollout)----
    for name, bid in scheme_ids.items():
        _, s = block_reduce(bid, {"vref": vref_tok})
        st["coh_num"][name] = st["coh_num"].get(name, 0.0) + float(np.abs(s["vref"]).sum())
        if name == "nl":
            _, cnt = block_reduce(bid, {"one": np.ones(bid.size)})
            st["nl_lens"].append(cnt["one"].astype(np.int64))
    st["coh_den"] += float(w_tok.sum())
    st["opd_num"] += float((d64 * i64).sum())
    st["opd_den"] += float(w_tok.sum())
    _accumulate_runs(ref, st)

    # ---- B1/B2:各陪审团的块级符号 ----
    if d2["size"].size == 0:
        return
    for size in np.unique(d2["size"]):
        sel = d2["size"] == size
        sub = {k: v[sel] for k, v in d2.items()}
        idx = _join(ref, sub)
        absd = np.abs(d64[idx])
        vref_t = absd * i64[idx]
        vhat_t = absd * sub["iota_hat"].astype(np.float64)
        opd_t = d64[idx] * i64[idx]
        w_t = np.abs(vref_t)
        jur = sub["jury"].astype(np.int64)
        for name, bid in scheme_ids.items():
            key = jur * BIG_JURY + bid[idx]
            nB, s = block_reduce(key, {
                "vhat": vhat_t, "vref": vref_t, "opd": opd_t,
                "phat": sub["iota_hat"].astype(np.float64), "pref": i64[idx]})
            e = _entry(st, size, name)
            wB = np.abs(s["vref"])
            valid = (s["vhat"] != 0) & (s["vref"] != 0)
            e["acc_num"] += float((wB * (valid & (np.sign(s["vhat"]) == np.sign(s["vref"])))).sum())
            e["acc_den"] += float(wB[valid].sum())
            pvalid = (s["phat"] != 0) & (s["pref"] != 0)
            e["pacc_num"] += float((wB * (pvalid & (np.sign(s["phat"]) == np.sign(s["pref"])))).sum())
            e["pacc_den"] += float(wB[pvalid].sum())
            e["real_num"] += float((np.sign(s["vhat"]) * s["vref"]).sum())
            e["tok_den"] += float(w_t.sum())
            e["n_blocks"] += nB
            if int(size) == train_g:
                c = st["cov"].setdefault(name, {"vhat": [], "vref": [], "opd": []})
                for k in ("vhat", "vref", "opd"):
                    c[k].append(s[k])
        if int(size) == train_g:
            st["cov_den"] += float(w_t.sum())


# ------------------------------------------------------------ 汇总
def _ratio(a, b):
    return a / b if b else float("nan")


def finalize(st, train_g):
    schemes = [f"T{t}" for t in FIXED_T] + (["nl"] if "nl" in st["coh_num"] else []) + ["roll"]
    sizes = sorted({s for s, _ in st["agg"]})
    summary = {"train_g": train_g, "sizes": sizes, "schemes": schemes,
               "opd_alignment": _ratio(st["opd_num"], st["opd_den"])}

    runs = np.concatenate(st["run_lens"]) if st["run_lens"] else np.array([1])
    b0 = {"adj_acc_unweighted": _ratio(st["adj_num_u"], st["adj_den_u"]),
          "adj_acc_w_mass": _ratio(st["adj_num_w"], st["adj_den_w"]),
          "run_len_mean": float(runs.mean()),
          "run_len_median": float(np.median(runs)),
          "run_len_tokweighted": _ratio(st["runlen_num_w"], st["runlen_den_w"]),
          "coherence": {k: _ratio(v, st["coh_den"]) for k, v in st["coh_num"].items()}}
    if st["nl_lens"]:
        nl = np.concatenate(st["nl_lens"])
        b0["nl_seg_len_mean"] = float(nl.mean())
        b0["nl_seg_len_median"] = float(np.median(nl))
    summary["b0"] = b0

    b1 = {}
    for name in schemes:
        b1[name] = {}
        for size in sizes:
            e = st["agg"].get((size, name))
            if e is None:
                continue
            b1[name][str(size)] = {
                "acc": _ratio(e["acc_num"], e["acc_den"]),
                "acc_plain_sum": _ratio(e["pacc_num"], e["pacc_den"]),
                "realized": _ratio(e["real_num"], e["tok_den"]),
                "n_blocks": e["n_blocks"]}
    summary["b1"] = b1

    b2 = {}
    for name, c in st["cov"].items():
        vhat = np.concatenate(c["vhat"])
        vref = np.concatenate(c["vref"])
        opd = np.concatenate(c["opd"])
        mag = np.abs(vhat)
        nz = mag > 0
        den = st["cov_den"]
        rows = {}
        for covr in COVERAGES:
            thr = float(np.quantile(mag[nz], 1 - covr)) if (covr < 1 and nz.any()) else 0.0
            keep = nz & (mag >= thr)
            valid = keep & (vref != 0)
            wB = np.abs(vref)
            acc = _ratio(float((wB * (valid & (np.sign(vhat) == np.sign(vref)))).sum()),
                         float(wB[valid].sum()))
            real_n = float((np.sign(vhat[keep]) * vref[keep]).sum())
            rows[f"{covr:.1f}"] = {
                "coverage_actual": _ratio(float(keep.sum()), float(nz.sum())),
                "acc": acc,
                "realized_neutral": _ratio(real_n, den),
                "realized_hybrid": _ratio(real_n + float(opd[~keep].sum()), den)}
        b2[name] = rows
    summary["b2"] = b2
    return summary


# ------------------------------------------------------------ 报告
def _best_scheme(summary, size):
    cand = [(summary["b1"][n][str(size)]["realized"], n) for n in summary["schemes"]
            if str(size) in summary["b1"].get(n, {})
            and summary["b1"][n][str(size)]["realized"] == summary["b1"][n][str(size)]["realized"]]
    return max(cand)[1] if cand else None


def write_report(summary, note, path):
    L = ["# 块级符号重分析(B0–B2)—— 结果报告\n",
         f"配置:{note};训练档 G={summary['train_g']};"
         f"块值 v_B=Σ|d|·ι(增益对齐),权重=|v_B^ref|,T=1 与 analysis.py 的 |d·ι| 口径一致。\n",
         f"**OPD 基线对齐率**(Σd·ι_ref/Σ|d·ι_ref|,'什么都不估、直接做标准 OPD'的一阶增益率):"
         f"**{summary['opd_alignment'] * 100:+.1f}%**\n"]
    b0 = summary["b0"]
    L.append("## B0 —— 真符号成片吗?(全部参考 rollout)\n")
    L.append(f"- 相邻 token 符号一致率:不加权 {_pct(b0['adj_acc_unweighted'])} · "
             f"质量加权 {_pct(b0['adj_acc_w_mass'])}(50% = 完全椒盐)")
    L.append(f"- 游程长度:均值 {b0['run_len_mean']:.1f} · 中位数 {b0['run_len_median']:.0f} · "
             f"质量加权所在游程 {b0['run_len_tokweighted']:.1f} token")
    if "nl_seg_len_mean" in b0:
        L.append(f"- 换行段长度:均值 {b0['nl_seg_len_mean']:.1f} · "
                 f"中位数 {b0['nl_seg_len_median']:.0f} token")
    L.append("- 块级证书保留率 coherence(最优块常值符号能保住的证书质量;1=无损):\n")
    names = summary["schemes"]
    L.append("| 方案 | " + " | ".join(SCHEME_LABEL[n] for n in names) + " |")
    L.append("|---|" + "---|" * len(names))
    L.append("| coherence | " + " | ".join(_pct(b0["coherence"].get(n)) for n in names) + " |")
    L.append("\n## B1 —— 块级符号准确率 / 实现增益率 vs (G × 块方案)\n")
    sizes = summary["sizes"]
    L.append("**符号准确率(权重 |v_B^ref|):**\n")
    L.append("| 方案 \\ G | " + " | ".join(str(s) for s in sizes) + " |")
    L.append("|---|" + "---|" * len(sizes))
    for n in names:
        L.append(f"| {SCHEME_LABEL[n]} | " + " | ".join(
            _pct(summary["b1"][n].get(str(s), {}).get("acc", float("nan"))) for s in sizes) + " |")
    L.append("\n**实现增益率(一阶增益 / token 级证书上限;OPD 基线 "
             f"{summary['opd_alignment'] * 100:+.1f}%):**\n")
    L.append("| 方案 \\ G | " + " | ".join(str(s) for s in sizes) + " |")
    L.append("|---|" + "---|" * len(sizes))
    for n in names:
        L.append(f"| {SCHEME_LABEL[n]} | " + " | ".join(
            f"{summary['b1'][n][str(s)]['realized'] * 100:+.1f}%"
            if str(s) in summary["b1"].get(n, {}) else "—" for s in sizes) + " |")

    tg = summary["train_g"]
    best = _best_scheme(summary, tg)
    L.append(f"\n## B2 —— 置信度–覆盖率(G={tg},按 |v_B^hat| 从高到低保留)\n")
    for n in ([f"T1"] + ([best] if best not in (None, "T1") else [])):
        if n not in summary["b2"]:
            continue
        L.append(f"**方案 {SCHEME_LABEL[n]}:**\n")
        L.append("| 目标覆盖率 | 实际覆盖率 | 保留块准确率 | 增益率(弃置=0) | 增益率(弃置回退 OPD) |")
        L.append("|---|---|---|---|---|")
        for covr, r in summary["b2"][n].items():
            L.append(f"| {float(covr) * 100:.0f}% | {_pct(r['coverage_actual'])} | {_pct(r['acc'])} | "
                     f"{r['realized_neutral'] * 100:+.1f}% | {r['realized_hybrid'] * 100:+.1f}% |")
        L.append("")

    # ---------------- 自动判读 ----------------
    L.append("## 判读(预写标准)\n")
    c8 = b0["coherence"].get("T8", float("nan"))
    adj = b0["adj_acc_w_mass"]
    if c8 == c8 and adj == adj:
        if adj >= 0.60 and c8 >= 0.75:
            L.append(f"- 相邻一致率 {_pct(adj)} ≥ 60% 且 coherence(T=8)={_pct(c8)} ≥ 75%:"
                     f"**真符号成片**,块常值符号几乎不丢证书质量,分块路线成立。")
        elif adj < 0.55:
            L.append(f"- 相邻一致率 {_pct(adj)} ≈ 50%(完全椒盐)⇒ **真符号不成片**,游程中位数 "
                     f"{b0['run_len_median']:.0f} token。coherence(T=8)={_pct(c8)} 偏高只是因为块内"
                     f"质量集中在个别重 token(重尾),不是符号连贯;分块靠'猜对主导 token'换准确率,"
                     f"证书质量仍按 coherence 衰减——最终以 B1 的实现增益率为准。")
        else:
            L.append(f"- 相邻一致率 {_pct(adj)}、coherence(T=8)={_pct(c8)}:真符号部分成片,"
                     f"分块有损;以 B1 的实现增益率为准。")
    if best == "T1":
        L.append("- **B1 增益率口径下没有任何块方案超过 T=1 ⇒ 分块提案否决**(符号信号就在 token "
                 "粒度上,块内求和丢证书质量快于换来的准确率)。")
    if best is not None:
        r_best = summary["b1"][best][str(tg)]["realized"]
        r_tok_gmax = summary["b1"]["T1"].get(str(max(sizes)), {}).get("realized", float("nan"))
        opd = summary["opd_alignment"]
        L.append(f"- G={tg} 最优方案 **{SCHEME_LABEL[best]}**:实现增益率 {r_best * 100:+.1f}%"
                 f"(逐 token G={max(sizes)} 为 {r_tok_gmax * 100:+.1f}%,OPD 基线 {opd * 100:+.1f}%)。")
        if r_best == r_best and r_tok_gmax == r_tok_gmax:
            if r_best >= r_tok_gmax:
                L.append(f"  ⇒ **G={tg}+分块 ≥ {max(sizes) // tg}× 开销的 G={max(sizes)} 逐 token**:分块是免费午餐,直接换。")
            elif r_best >= 0.6 * r_tok_gmax:
                L.append(f"  ⇒ G={tg}+分块拿到 G={max(sizes)} 逐 token 的 {r_best / r_tok_gmax * 100:.0f}%,但开销只有 1/{max(sizes) // tg}:性价比在分块一侧。")
            else:
                L.append(f"  ⇒ 分块也追不回大陪审团:估计器本身(投影/留一/参考)还有别的问题,回到 F1 修估计器。")
        if r_best == r_best and opd == opd and r_best <= opd:
            L.append(f"  ⚠ 最优方案的增益率未超过 OPD 基线({opd * 100:+.1f}%):在这个估计精度下,'直接做 OPD'确实更划算——分块+置信门控(见 B2)是否翻盘看下一条。")
        if best in summary["b2"]:
            hyb = {c: r["realized_hybrid"] for c, r in summary["b2"][best].items()}
            cbest = max(hyb, key=lambda c: hyb[c] if hyb[c] == hyb[c] else -9)
            L.append(f"- 推荐配置:方案 {SCHEME_LABEL[best]} + 覆盖率 {float(cbest) * 100:.0f}%"
                     f"(弃置块回退标准 OPD),增益率 {hyb[cbest] * 100:+.1f}%——超过 OPD 基线"
                     f" {(hyb[cbest] - opd) * 100:+.1f}pp 才值得上 targeted。")
    L.append("- 注意:参考半批只存了符号,块级参考自稳定性算不出来;块内求和对参考噪声同样有 "
             "√T 平均效应,故块级绝对数值比 token 级报告更可信,横向比较不受影响。")
    L.append("\n*由 probe/block_analysis.py 自动生成;原始数据 q*/tokens.npz、d2.npz。*")
    with open(path, "w") as f:
        f.write("\n".join(L))


# ------------------------------------------------------------ 入口
def _newline_flags(y, tok, cache):
    out = np.zeros(y.size, bool)
    uniq = np.unique(y)
    for t in uniq:
        t = int(t)
        if t not in cache:
            cache[t] = "\n" in tok.decode([t])
    nl_ids = np.array([t for t in uniq if cache[int(t)]], dtype=y.dtype)
    if nl_ids.size:
        out = np.isin(y, nl_ids)
    return out


def run_block_analysis(cfg, train_g=None):
    manifest_path = os.path.join(cfg.out_dir, "manifest.json")
    student = cfg.student
    note = f"dataset={cfg.dataset}"
    if os.path.exists(manifest_path):        # 以落盘 manifest 为准,防 F9 配置漂移
        m = json.load(open(manifest_path))
        student = m.get("student", student)
        note = (f"dataset={m.get('dataset')} R={m.get('rollouts_per_question')} "
                f"cap={m.get('max_new_tokens')} adapter={m.get('adapter_path')}")
    qids = [q["idx"] for q in json.load(open(os.path.join(cfg.out_dir, "questions.json")))]

    tok, cache = None, {}
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(student)
    except Exception as e:  # noqa: BLE001
        print(f"[block] ⚠ tokenizer 加载失败({e}),跳过换行分段方案")

    st = new_state()
    sizes_seen = set()
    for qi in qids:
        ref = dict(np.load(os.path.join(cfg.out_dir, f"q{qi}", "tokens.npz")))
        d2 = dict(np.load(os.path.join(cfg.out_dir, f"q{qi}", "d2.npz")))
        if d2["size"].size:
            sizes_seen.update(int(s) for s in np.unique(d2["size"]))
        is_nl = _newline_flags(ref["y"], tok, cache) if tok is not None else None
        if train_g is None:
            train_g = min(sizes_seen) if sizes_seen else 8
        analyze_question(ref, d2, is_nl, train_g, st)
        print(f"[block] q{qi} 完成({len(ref['pos'])} token)")

    summary = finalize(st, train_g)
    res_dir = os.path.join(cfg.out_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, "block_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    write_report(summary, note, os.path.join(res_dir, "block_report.md"))
    print(f"[block] 完成 → {res_dir}/block_summary.json, block_report.md")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--train-g", type=int, default=None,
                    help="训练档陪审团大小(默认取数据里最小的档,即 8)")
    args = ap.parse_args()
    from .config import load_config
    cfg = load_config(args.config, out_dir=args.out)
    run_block_analysis(cfg, train_g=args.train_g)


if __name__ == "__main__":
    main()
