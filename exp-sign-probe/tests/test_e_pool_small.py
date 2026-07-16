# -*- coding: utf-8 -*-
"""E1/E2 探针的暴力对照单测(CPU 秒级)。
运行:python tests/test_e_pool_small.py"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from probe.e_pool_head import (_MLP, _Logistic, combine_conditions, draw_jury,
                               masswise_metrics, pick_indices, score_cached)
from probe.influence import brute_iota

V, HD = 40, 16


def test_score_cached_matches_brute_and_linear():
    g = torch.Generator().manual_seed(0)
    n = 37
    G1 = torch.randn(V, HD, generator=g)
    G2 = torch.randn(V, HD, generator=g)
    h = torch.randn(n, HD, generator=g)
    pi = torch.softmax(torch.randn(n, V, generator=g), -1)
    y = torch.randint(0, V, (n,), generator=g)
    s1 = score_cached(G1, h.half().numpy(), pi.half().numpy(), y.numpy(), "cpu")
    s12 = score_cached(G1 + G2, h.half().numpy(), pi.half().numpy(), y.numpy(), "cpu")
    s2 = score_cached(G2, h.half().numpy(), pi.half().numpy(), y.numpy(), "cpu")
    for t in range(n):
        assert abs(s1[t] - brute_iota(G1, pi[t], h[t], y[t])) < 5e-2   # fp16 缓存容差
    assert np.allclose(s12, s1 + s2, atol=1e-1)                        # 线性性


def test_combine_conditions_pairing():
    K, S, D, n = 5, 3, 2, 4
    own = np.arange(K * n, dtype=np.float32).reshape(K, n)
    oth8 = np.ones((S, D, n), np.float32)
    oth8[:, 1] = 2.0
    oth32 = 10 * oth8
    c = combine_conditions(own, oth8, oth32)
    assert np.allclose(c["C0"], own)
    # 第 k 行配第 k%D 组:k=0 → draw0(和=S·1),k=1 → draw1(和=S·2)
    assert np.allclose(c["C1"][0], own[0] + S * 1.0)
    assert np.allclose(c["C1"][1], own[1] + S * 2.0)
    assert np.allclose(c["C2"][0], own[0] + S * 10.0)
    assert np.allclose(c["C2m"][0], own[0] + 0.25 * S * 10.0)
    assert np.allclose(c["C3"][1], np.full(n, S * 20.0))


def test_draw_jury_and_pick_indices():
    rng = np.random.default_rng(0)
    R = 256
    ev, hd_, pool = pick_indices(R, rng)
    assert len(ev) == 32 and len(hd_) == 32 and len(pool) == R - 64
    assert not (set(ev) & set(hd_)) and not (set(ev) | set(hd_)) & set(pool)
    verdicts = np.zeros(R, np.float32)
    verdicts[pool[:3]] = 1.0                      # 池内有混合
    members, adv = draw_jury(rng, pool, 8, verdicts)
    assert members is not None and set(members) <= set(pool.tolist())
    assert abs(adv[np.array(members)].sum()) < 1e-6          # 组内中心化
    none_m, _ = draw_jury(rng, pool, 8, np.zeros(R, np.float32))
    assert none_m is None                                    # 全错 ⇒ 退化


def test_masswise_metrics_extremes():
    s_ref = np.array([1, -1, 1, -1], np.float32)
    w = np.array([1, 2, 3, 4], np.float32)
    acc, real = masswise_metrics(s_ref.copy(), s_ref, w)
    assert abs(acc - 1) < 1e-9 and abs(real - 1) < 1e-9
    acc, real = masswise_metrics(-s_ref, s_ref, w)
    assert abs(acc) < 1e-9 and abs(real + 1) < 1e-9


def test_head_models_forward():
    h = torch.randn(10, HD)
    y = torch.randint(0, V, (10,))
    sc = torch.randn(10, 2)
    assert _MLP(HD, V)(h, y, sc).shape == (10, 1)
    assert _MLP(HD, V, use_h=False)(h, y, sc).shape == (10, 1)
    assert _Logistic(HD)(h, y, sc).shape == (10, 1)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"[ok] {fn.__name__}")
    print("全部通过")
