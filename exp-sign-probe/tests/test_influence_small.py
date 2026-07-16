# -*- coding: utf-8 -*-
"""D1-lite:小词表暴力对照,验证 influence.py 的三个恒等式(CPU,秒级)。
运行:python -m pytest tests/ -q   或   python tests/test_influence_small.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from probe.influence import (accumulate_G, brute_G, brute_iota, iota_from_G,
                             own_correction)

V, HD = 50, 8


def fake_rollouts(R=6, Tmin=3, Tmax=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    rolls = []
    for _ in range(R):
        T = int(torch.randint(Tmin, Tmax + 1, (1,), generator=g))
        logits = torch.randn(T, V, generator=g)
        rolls.append({
            "probs": torch.softmax(logits, dim=-1),
            "h": torch.randn(T, HD, generator=g),
            "y": torch.randint(0, V, (T,), generator=g),
        })
    verdicts = torch.tensor([1., 0., 1., 1., 0., 0.][:R])
    adv = verdicts - verdicts.mean()
    return rolls, adv


def build_G_vectorized(rolls, adv, member_idx):
    G = torch.zeros(V, HD)
    for ri in member_idx:
        r = rolls[ri]
        a = torch.full((r["y"].shape[0],), float(adv[ri]))
        accumulate_G(G, r["probs"], r["h"], r["y"], a)
    return G


def build_G_brute(rolls, adv, member_idx):
    G = torch.zeros(V, HD)
    for ri in member_idx:
        r = rolls[ri]
        a = torch.full((r["y"].shape[0],), float(adv[ri]))
        G += brute_G(r["probs"], r["h"], r["y"], a)
    return G


def test_G_accumulation_matches_brute():
    rolls, adv = fake_rollouts()
    Gv = build_G_vectorized(rolls, adv, list(range(len(rolls))))
    Gb = build_G_brute(rolls, adv, list(range(len(rolls))))
    assert torch.allclose(Gv, Gb, atol=1e-5), (Gv - Gb).abs().max()


def test_iota_matches_brute():
    rolls, adv = fake_rollouts()
    G = build_G_brute(rolls, adv, list(range(len(rolls))))
    r = rolls[2]
    vec = iota_from_G(G, r["probs"], r["h"], r["y"])
    for t in range(r["y"].shape[0]):
        b = brute_iota(G, r["probs"][t], r["h"][t], int(r["y"][t]))
        assert abs(float(vec[t]) - b) < 1e-4


def test_loo_identity():
    """全批 G 打分 − a_j·own_correction == 去掉 rollout j 后重建 G 再打分(精确留一)。"""
    rolls, adv = fake_rollouts()
    all_idx = list(range(len(rolls)))
    G_full = build_G_vectorized(rolls, adv, all_idx)
    for j in [0, 3]:
        r = rolls[j]
        loo = (iota_from_G(G_full, r["probs"], r["h"], r["y"])
               - float(adv[j]) * own_correction(r["probs"], r["h"], r["y"]))
        G_wo = build_G_vectorized(rolls, adv, [i for i in all_idx if i != j])
        direct = iota_from_G(G_wo, r["probs"], r["h"], r["y"])
        assert torch.allclose(loo, direct, atol=1e-4), (loo - direct).abs().max()


def test_halves_sum_to_full():
    rolls, adv = fake_rollouts()
    member_idx = list(range(len(rolls)))
    h0 = [ri for k, ri in enumerate(member_idx) if k % 2 == 0]
    h1 = [ri for k, ri in enumerate(member_idx) if k % 2 == 1]
    G0 = build_G_vectorized(rolls, adv, h0)
    G1 = build_G_vectorized(rolls, adv, h1)
    Gf = build_G_vectorized(rolls, adv, member_idx)
    assert torch.allclose(G0 + G1, Gf, atol=1e-5)


def test_degenerate_group_gives_zero():
    rolls, _ = fake_rollouts()
    adv = torch.zeros(len(rolls))     # 全对或全错 ⇒ A≡0
    G = build_G_vectorized(rolls, adv, list(range(len(rolls))))
    assert float(G.abs().max()) == 0.0


if __name__ == "__main__":
    for fn in [test_G_accumulation_matches_brute, test_iota_matches_brute,
               test_loo_identity, test_halves_sum_to_full,
               test_degenerate_group_gives_zero]:
        fn()
        print(f"PASS  {fn.__name__}")
    print("influence.py 全部恒等式通过")
