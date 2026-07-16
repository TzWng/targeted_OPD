# -*- coding: utf-8 -*-
"""软权重 / ε-拒绝域 奖励方案的离线重放(零采样,只读已有 d2.npz + tokens.npz)。

问题:把硬符号 r=sign(ι̂)|d| 换成软权重或 ε-拒绝,能否提高实现增益率?
理论预期(预写):盒约束 |r|≤|d| 下,对称校准噪声时硬符号逐步条件最优;
软/ε 的收益只可能来自 (a) 极小 |ι̂| 区的反校准,(b) 拒绝后回退教师 d_t。

指标:实现增益率 RG = Σ_事件 r_t·ι*_t / Σ_事件 |d_t·ι*_t| ∈ [−1, 1]
(与块级报告同口径;oracle=100%,OPD 基线 dm≈+8.5% / gsm8k≈+31.2%,
 硬符号@m=8 应复现块报告 T=1 行 ≈+17.3% / +29.0% —— 三个运行时锚点)。

方案(ε、τ 均为该陪审团成员 token 内 |ι̂| 的分位数,跨 m 尺度不变):
  opd            r = d_t(什么都不估)
  oracle         r = sign(ι*)|d|(上限)
  hard           r = sign(ι̂)|d|(论文式 12)
  gate_current   两票一致→sign(ι̂)|d|,否则→d_t(训练现行方案,用 alt_sign)
  dead0_pXX      |ι̂| ≥ P_XX → sign(ι̂)|d|,否则 → 0(论文中性桶 + ε)
  deadT_pXX      |ι̂| ≥ P_XX → sign(ι̂)|d|,否则 → d_t(ε-拒绝回退教师)
  soft_pXX       r = clip(ι̂/τ, −1, 1)·|d|,τ = P_XX(软线性)
  softdeadT      |ι̂| ≥ P30 → clip(ι̂/P70)|d|,否则 → d_t(软 + 拒绝组合)
输出:<run-dir>/results/soft_reward_report.md + soft_reward_summary.json。"""
import argparse
import json
import os

import numpy as np

KEY = 1 << 14


def key(r, p):
    return r.astype(np.int64) * KEY + p.astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--eps", type=int, nargs="+", default=[20, 40, 60])
    ap.add_argument("--taus", type=int, nargs="+", default=[50, 70, 90])
    args = ap.parse_args()
    run = args.run_dir
    qids = [m["idx"] for m in json.load(open(os.path.join(run, "questions.json")))]

    schemes = (["opd", "oracle", "hard", "gate_current"]
               + [f"dead0_p{e}" for e in args.eps]
               + [f"deadT_p{e}" for e in args.eps]
               + [f"soft_p{t}" for t in args.taus]
               + ["softdeadT"])
    sums = {}          # (scheme, size) -> [num, den]

    def acc(scheme, size, num, den):
        k = (scheme, size)
        if k not in sums:
            sums[k] = [0.0, 0.0]
        sums[k][0] += num
        sums[k][1] += den

    sizes_seen = set()
    for qi in qids:
        tk = np.load(os.path.join(run, f"q{qi}", "tokens.npz"))
        d2 = np.load(os.path.join(run, f"q{qi}", "d2.npz"))
        if d2["size"].size == 0:
            continue
        rk = key(tk["rollout"], tk["pos"])
        order = np.argsort(rk)
        idx = order[np.searchsorted(rk[order], key(d2["rollout"], d2["pos"]))]
        istar = tk["ref_iota"][idx].astype(np.float64)
        dmag = np.abs(tk["d"][idx].astype(np.float64))
        dsgn = np.sign(tk["d"][idx].astype(np.float64))
        ih = d2["iota_hat"].astype(np.float64)
        alt = d2["alt_sign"].astype(np.float64)
        den_tok = np.abs(dmag * istar)
        d_gain = dmag * dsgn * istar                 # OPD 的逐事件贡献 d·ι*
        s_gain = dmag * np.sign(ih) * istar          # 硬符号贡献
        o_gain = dmag * np.sign(istar) * istar       # oracle
        gate = (np.sign(ih) == alt) & (alt != 0)
        g_gain = np.where(gate, s_gain, d_gain)      # 训练现行门控

        # 逐(size, jury)算分位数尺度
        sz = d2["size"].astype(int)
        ju = d2["jury"].astype(int)
        gid = sz * 10000 + ju
        for g in np.unique(gid):
            m = gid == g
            size = int(sz[m][0])
            sizes_seen.add(size)
            a = np.abs(ih[m])
            den = float(den_tok[m].sum())
            acc("opd", size, float(d_gain[m].sum()), den)
            acc("oracle", size, float(o_gain[m].sum()), den)
            acc("hard", size, float(s_gain[m].sum()), den)
            acc("gate_current", size, float(g_gain[m].sum()), den)
            qs = {p: np.percentile(a, p) for p in
                  set(args.eps) | set(args.taus) | {30, 70}}
            for e in args.eps:
                keep = a >= qs[e]
                acc(f"dead0_p{e}", size,
                    float(np.where(keep, s_gain[m], 0.0).sum()), den)
                acc(f"deadT_p{e}", size,
                    float(np.where(keep, s_gain[m], d_gain[m]).sum()), den)
            for t in args.taus:
                w = np.clip(ih[m] / max(qs[t], 1e-12), -1.0, 1.0)
                acc(f"soft_p{t}", size,
                    float((dmag[m] * w * istar[m]).sum()), den)
            keep = a >= qs[30]
            w = np.clip(ih[m] / max(qs[70], 1e-12), -1.0, 1.0)
            soft = dmag[m] * w * istar[m]
            acc("softdeadT", size,
                float(np.where(keep, soft, d_gain[m]).sum()), den)
        print(f"[soft] q{qi} 完成({len(ih):,} 事件)")

    sizes = sorted(sizes_seen)
    table = {s: {str(m): (sums[(s, m)][0] / sums[(s, m)][1]
                          if (s, m) in sums and sums[(s, m)][1] else float("nan"))
                 for m in sizes} for s in schemes}
    out = {"run_dir": run, "sizes": sizes, "rg": table}
    os.makedirs(os.path.join(run, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(run, "results",
                                     "soft_reward_summary.json"), "w"), indent=1)

    P = lambda x: "—" if x != x else f"{100 * x:+.1f}%"
    L = ["# 软权重 / ε-拒绝域 —— 离线重放报告\n",
         f"数据:{run};指标 = 实现增益率 RG(与块级报告同口径,oracle=100%)。",
         "ε/τ 为陪审团内 |ι̂| 分位数;`deadT` = 拒绝后回退教师;`dead0` = 拒绝后置零(论文中性桶)。\n",
         "| 方案 \\ m | " + " | ".join(str(m) for m in sizes) + " |",
         "|---|" + "---|" * len(sizes)]
    for s in schemes:
        L.append(f"| {s} | " + " | ".join(P(table[s][str(m)]) for m in sizes) + " |")
    m0 = str(sizes[0])
    hard0, opd0 = table["hard"][m0], table["opd"][m0]
    best = max(((s, table[s][m0]) for s in schemes
                if s not in ("oracle",)), key=lambda x: x[1])
    L.append(f"\n## 判读(m={m0},训练档)\n")
    L.append(f"- 锚点自检:hard 应≈块级报告 T=1 行;opd 应≈块级 OPD 基线。")
    L.append(f"- 最优方案:**{best[0]}**,RG {P(best[1])}(hard {P(hard0)},opd {P(opd0)})。")
    if best[1] <= max(hard0, opd0) + 0.01:
        L.append("- **结论:软权重/ε-拒绝没有带来实质提升**——与理论一致(盒约束下硬符号是"
                 "逐步条件最优;|ι̂| 又几乎无校准信息),收益只剩拒绝回退教师的部分,幅度有限。"
                 "瓶颈仍是 ĝ_Q 的有效样本量。")
    else:
        L.append(f"- **结论:{best[0]} 相对 hard 提升 {P(best[1] - hard0)}**"
                 "——值得作为训练臂纳入(注意它相对 OPD 的余量才是部署判据)。")
    open(os.path.join(run, "results", "soft_reward_report.md"), "w").write("\n".join(L))
    print("[soft] 完成 →", os.path.join(run, "results", "soft_reward_report.md"))


if __name__ == "__main__":
    main()
