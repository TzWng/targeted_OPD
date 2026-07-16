# -*- coding: utf-8 -*-
"""D4 核心恒等式的小模型验证(纯 torch,CPU 秒级,不需要 transformers):
 1. K[i,t] = <u_i, g_t> 线性分解 ⇒ 任意陪审团 ι̂ = Σ A_i K[i,t](对照直接 <ĝ_Q, g_t>);
 2. K 空间的留一 == 真的把 rollout j 从陪审团里拿掉重算;
 3. lm_head 闭式 Gram 核 == autograd 算出的 lm_head 子空间核。
运行:python tests/test_d4_small.py"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

V, H, RANK = 40, 12, 3
g = torch.Generator().manual_seed(0)


class TinyLM(torch.nn.Module):
    """embed → (linear+tanh, 带 LoRA A/B) → lm_head。B=0 初始化,和真实设置一致。"""
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(V, H)
        self.mid = torch.nn.Linear(H, H, bias=False)
        self.lora_A = torch.nn.Parameter(torch.randn(RANK, H, generator=g) * 0.3)
        self.lora_B = torch.nn.Parameter(torch.zeros(H, RANK))
        self.lm_head = torch.nn.Linear(H, V, bias=False)

    def hidden(self, ids):                        # [L] -> [L, H]
        x = self.emb(ids)
        x = torch.tanh(self.mid(x) + x @ self.lora_A.t() @ self.lora_B.t())
        return x

    def token_logps(self, ids, plen):
        h = self.hidden(ids)[plen - 1:-1]
        logits = self.lm_head(h)
        y = ids[plen:]
        return F.log_softmax(logits, dim=-1).gather(1, y.unsqueeze(1)).squeeze(1), h, y


def make_rollouts(R=10, plen=3, tmin=4, tmax=8):
    rolls = []
    for _ in range(R):
        L = plen + int(torch.randint(tmin, tmax + 1, (1,), generator=g))
        rolls.append(torch.randint(0, V, (L,), generator=g))
    verd = torch.tensor([1., 0., 1., 0., 0., 1., 0., 0., 1., 0.][:R])
    return rolls, verd


def grads_in(model, scalar, params):
    return torch.cat([x.reshape(-1) for x in
                      torch.autograd.grad(scalar, params, retain_graph=True)])


def build_K(model, rolls, plen, params):
    """K[i, (j,t)] = <u_i, g_{j,t}>,列 = 所有 completion token(全采样)。"""
    us, gs, owner = [], [], []
    for j, ids in enumerate(rolls):
        logps, _, _ = model.token_logps(ids, plen)
        us.append(grads_in(model, logps.sum(), params))
        for t in range(logps.shape[0]):
            gs.append(grads_in(model, logps[t], params))
            owner.append(j)
    U = torch.stack(us)
    Gm = torch.stack(gs)
    return U @ Gm.t(), torch.tensor(owner), Gm, U


def test_jury_linearity_and_loo():
    model = TinyLM()
    rolls, verd = make_rollouts()
    plen = 3
    params = [model.lora_A, model.lora_B]
    K, owner, Gm, U = build_K(model, rolls, plen, params)
    mem = [0, 2, 3, 7]                            # 混合判分的陪审团
    a = torch.zeros(len(rolls))
    a[mem] = verd[mem] - verd[mem].mean()
    # 1) 线性分解 = 直接 <ĝ_Q, g_t>
    gQ = (a.unsqueeze(1) * U).sum(0)
    direct = Gm @ gQ
    viaK = K.t() @ a
    assert torch.allclose(direct, viaK, atol=1e-5), (direct - viaK).abs().max()
    # 2) K 空间留一 == 重建无 j 陪审团
    j = 2
    est_loo = viaK - a[j] * K[j]                  # 只对 t∈j 有意义,全列算也应相等
    a_wo = a.clone(); a_wo[j] = 0.0
    direct_wo = Gm @ ((a_wo.unsqueeze(1) * U).sum(0))
    tok_j = owner == j
    assert torch.allclose(est_loo[tok_j], direct_wo[tok_j], atol=1e-5)
    print("PASS  jury 线性分解 + K 空间留一(LoRA 子空间)")


def test_lm_closed_form_kernel():
    model = TinyLM()
    rolls, verd = make_rollouts()
    plen = 3
    params = [model.lm_head.weight]
    K_auto, owner, _, _ = build_K(model, rolls, plen, params)
    # 闭式:K[i,t] = Σ_{s∈i} <h_s,h_t>[EQ − π_s(y_t) − π_t(y_s) + <π_s,π_t>]
    Hs, Ps, Ys = [], [], []
    for ids in rolls:
        with torch.no_grad():
            logps, h, y = model.token_logps(ids, plen)
            pr = F.softmax(model.lm_head(h), dim=-1)
        Hs.append(h); Ps.append(pr); Ys.append(y)
    Hall = torch.cat(Hs); Pall = torch.cat(Ps); Yall = torch.cat(Ys)
    n = Hall.shape[0]
    K_cf = torch.zeros(len(rolls), n)
    for i in range(len(rolls)):
        C = Hs[i] @ Hall.t()
        M1 = Ps[i][:, Yall]
        PT = Pall[:, Ys[i]].t()
        PP = Ps[i] @ Pall.t()
        EQ = (Ys[i].unsqueeze(1) == Yall.unsqueeze(0)).float()
        K_cf[i] = (C * (EQ - M1 - PT + PP)).sum(0)
    assert torch.allclose(K_auto, K_cf, atol=1e-4), (K_auto - K_cf).abs().max()
    print("PASS  lm_head 闭式 Gram 核 == autograd 核")


def test_lora_grad_lives_in_B_at_init():
    model = TinyLM()
    rolls, _ = make_rollouts()
    logps, _, _ = model.token_logps(rolls[0], 3)
    gA, gB = torch.autograd.grad(logps.sum(), [model.lora_A, model.lora_B])
    assert float(gA.abs().max()) < 1e-9 and float(gB.abs().max()) > 0
    print("PASS  B=0 初始化下梯度集中于 B(训练第 0 步所见)")


if __name__ == "__main__":
    test_jury_linearity_and_loo()
    test_lm_closed_form_kernel()
    test_lora_grad_lives_in_B_at_init()
    print("D4 全部恒等式通过")
