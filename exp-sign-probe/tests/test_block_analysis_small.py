# -*- coding: utf-8 -*-
"""块级重分析的暴力对照单测(纯 numpy,CPU 秒级)。
运行:python -m pytest tests/test_block_analysis_small.py -q
     或 python tests/test_block_analysis_small.py

植入真值的合成数据:真符号按 8-token 游程排列(与 T=8 窗口对齐),
估计 = 真值 + 大噪声。应恢复:
  * coherence(T=8) = 1(块与游程对齐,块常值符号无损);
  * 块级(T=8)准确率与实现增益率 > 逐 token(T=1);
  * 零噪声时逐 token 准确率 = 实现增益率 = 1;
  * block_reduce / 换行分段与暴力循环一致。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from probe.block_analysis import (analyze_question, block_reduce, finalize,
                                  fixed_block_ids, new_state,
                                  newline_block_ids)


# ------------------------------------------------------------ 工具
def make_ref(R=6, T=64, run=8, seed=0):
    """真 ι 按 run 长度成片(±交替),|d|=均匀正数 ⇒ coherence(T=run)=1。"""
    g = np.random.default_rng(seed)
    rollout, pos, iota, d = [], [], [], []
    for r in range(R):
        s0 = 1 if r % 2 == 0 else -1
        signs = np.repeat([s0 * (-1) ** k for k in range(T // run)], run)
        mag = g.uniform(0.5, 1.5, T)
        rollout.append(np.full(T, r, np.int32))
        pos.append(np.arange(T, dtype=np.int32))
        iota.append(signs * mag)
        d.append(g.uniform(0.2, 1.0, T) * g.choice([-1, 1], T))
    return {"rollout": np.concatenate(rollout), "pos": np.concatenate(pos),
            "ref_iota": np.concatenate(iota).astype(np.float32),
            "d": np.concatenate(d).astype(np.float32),
            "y": np.zeros(R * T, np.int32)}


def make_d2(ref, sigma, juries=8, size=8, seed=1):
    """每个陪审团覆盖全部 rollout,iota_hat = ref_iota + N(0, σ²)。"""
    g = np.random.default_rng(seed)
    n = ref["pos"].size
    cols = {k: [] for k in ("size", "jury", "rollout", "pos", "iota_hat", "alt_sign")}
    for k in range(juries):
        cols["size"].append(np.full(n, size, np.int16))
        cols["jury"].append(np.full(n, k, np.int16))
        cols["rollout"].append(ref["rollout"])
        cols["pos"].append(ref["pos"])
        cols["iota_hat"].append((ref["ref_iota"] + g.normal(0, sigma, n)).astype(np.float32))
        cols["alt_sign"].append(np.zeros(n, np.int8))
    return {k: np.concatenate(v) for k, v in cols.items()}


def run_pipeline(ref, d2, train_g=8):
    st = new_state()
    analyze_question(ref, d2, None, train_g, st)
    return finalize(st, train_g)


# ------------------------------------------------------------ 测试
def test_block_reduce_matches_brute():
    g = np.random.default_rng(0)
    key = g.integers(0, 20, 500)
    val = g.normal(size=500)
    n, s = block_reduce(key, {"v": val})
    brute = {k: val[key == k].sum() for k in np.unique(key)}
    assert n == len(brute)
    assert np.allclose(s["v"], [brute[k] for k in sorted(brute)], atol=1e-10)


def test_newline_segments_brute():
    rollout = np.array([0, 0, 0, 0, 0, 1, 1, 1], np.int32)
    pos = np.array([0, 1, 2, 3, 4, 0, 1, 2], np.int32)
    is_nl = np.array([0, 1, 0, 0, 1, 0, 0, 0], bool)   # 换行结束一段
    seg = newline_block_ids(rollout, pos, is_nl)
    # rollout0: [0,1] [2,3,4] · rollout1: [5,6,7];段 id 不跨 rollout
    assert seg[0] == seg[1] and seg[2] == seg[3] == seg[4] and seg[5] == seg[6] == seg[7]
    assert len({seg[0], seg[2], seg[5]}) == 3


def test_fixed_blocks_do_not_cross_rollouts():
    rollout = np.array([0, 0, 1, 1], np.int32)
    pos = np.array([0, 1, 0, 1], np.int32)
    bid = fixed_block_ids(rollout, pos, 4)
    assert bid[0] == bid[1] and bid[2] == bid[3] and bid[0] != bid[2]


def test_perfect_estimator_is_exact():
    ref = make_ref()
    s = run_pipeline(ref, make_d2(ref, sigma=0.0))
    r = s["b1"]["T1"]["8"]
    assert abs(r["acc"] - 1.0) < 1e-9
    assert abs(r["realized"] - 1.0) < 1e-9


def test_coherence_one_when_runs_align():
    ref = make_ref(run=8)
    s = run_pipeline(ref, make_d2(ref, sigma=0.0))
    assert abs(s["b0"]["coherence"]["T8"] - 1.0) < 1e-9      # 与游程对齐 ⇒ 无损
    assert s["b0"]["coherence"]["T16"] < 0.7                 # 跨游程 ⇒ 有损
    assert abs(s["b0"]["run_len_mean"] - 8.0) < 1e-9


def test_blocks_beat_tokens_under_noise():
    ref = make_ref(R=24, T=64, run=8, seed=2)
    s = run_pipeline(ref, make_d2(ref, sigma=3.0, juries=8, seed=3))
    t1, t8 = s["b1"]["T1"]["8"], s["b1"]["T8"]["8"]
    assert t8["acc"] > t1["acc"] + 0.05, (t1["acc"], t8["acc"])
    assert t8["realized"] > t1["realized"], (t1["realized"], t8["realized"])
    # 全覆盖时 T=1 的增益率 ≈ 2·acc−1(同一质量口径)
    assert abs(t1["realized"] - (2 * t1["acc"] - 1)) < 1e-6


def test_coverage_gating_monotone_bookkeeping():
    ref = make_ref(R=24, T=64, run=8, seed=4)
    s = run_pipeline(ref, make_d2(ref, sigma=3.0, juries=8, seed=5))
    rows = s["b2"]["T8"]
    full = rows["1.0"]
    assert abs(full["coverage_actual"] - 1.0) < 1e-9
    assert abs(full["realized_neutral"] - s["b1"]["T8"]["8"]["realized"]) < 1e-9
    top = rows["0.2"]
    assert top["coverage_actual"] < 0.35
    assert top["acc"] >= full["acc"] - 1e-9   # 高置信块不应更差(合成数据下应更好)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"[ok] {fn.__name__}")
    print("全部通过")
