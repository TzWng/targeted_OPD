# -*- coding: utf-8 -*-
"""D4:投影子空间对照 —— lm_head(现状)vs LoRA 块(训练真正移动的子空间)。

设计(复用已完成探针的 rollout 与参考,零重新采样):
  任意子空间 P 下,陪审团估计线性分解为核矩阵:
      ι̂_J(t) = Σ_{i∈J} A_i^{(J)} K_P[i,t] − A_{j(t)}^{(J)} K_P[j(t),t](留一,t ∈ rollout j)
  其中 K_P[i,t] = ⟨u_i^P, g_t^P⟩:
      u_i = rollout i 全部 completion token 的 Σlogp 的 P-梯度(一条一次反传);
      g_t = 单个 token logp 的 P-梯度(每个抽样 token 一次反传)。
  * LoRA 侧:autograd(基座冻结;全新 LoRA B=0 ⇒ 梯度集中于 B,恰为训练第 0 步真实所见);
  * lm_head 侧:g_t = (e_y − π_t)h_tᵀ 闭式 ⇒ K 由 Gram 恒等式矩阵乘直接得到,
    并与已存 tokens.npz 的 ref_iota 对账(运行时自检,应 ≈100% 同号)。
之后一切(参考、稳定性、陪审团扫描、留一)都是 K 上的 numpy;两个子空间共用同一批陪审团 ⇒ 逐 token 配对。
输出:<run-dir>/d4/q*.npz + d4_summary.json + d4_report.md(预写判读)。"""
import argparse
import gc
import json
import os
import types

import numpy as np
import torch
import torch.nn.functional as F

from .rollout import load_question
from .stages import rollout_tensors

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


# ------------------------------------------------------------------ 通用
def cfg_from_manifest(run_dir):
    m = json.load(open(os.path.join(run_dir, "manifest.json")))
    return types.SimpleNamespace(out_dir=run_dir, **{k: m[k] for k in
        ("student", "adapter_path", "include_truncated", "dataset")})


def load_student_with_lora(cfg, r, alpha, seed, device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.student)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        cfg.student, torch_dtype=dtype, attn_implementation="sdpa").to(device)
    torch.manual_seed(seed)                       # LoRA 初始化可复现
    if cfg.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg.adapter_path,
                                          is_trainable=True)
        print(f"[d4] 使用训练中期适配器: {cfg.adapter_path}")
    else:
        lconf = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0,
                           target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        model = get_peft_model(model, lconf)
        print(f"[d4] 挂载全新 LoRA(r={r}, α={alpha}, seed={seed};"
              f"B=0 ⇒ 前向不变、梯度=训练第 0 步所见)")
    model.eval()
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[d4] 可训练(LoRA)参数维度: {n_par:,}")
    return tok, model


def lora_params(model):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    named.sort(key=lambda x: x[0])
    return [p for _, p in named]


def flat_grads(grads):
    return torch.cat([g.reshape(-1).float() for g in grads])


def token_logps(model, ids, plen, device):
    """一次前向(带图):completion 每个位置的 logp [T](保留计算图)。"""
    ids = ids.to(device).unsqueeze(0)
    out = model(input_ids=ids)
    logits = out.logits[0, plen - 1:-1, :]        # 位置 j 预测 ids[j+1]
    y = ids[0, plen:]
    return F.log_softmax(logits.float(), dim=-1).gather(
        1, y.unsqueeze(1)).squeeze(1)


# ----------------------------------------------------------- LoRA 侧核矩阵
def lora_kernel_for_question(model, rts, samp_by_roll, n_samp, device,
                             log_every=32):
    """K_lora[R, n_samp](fp32, CPU)。samp_by_roll: {rollout: [(样本序号, 位置)]}"""
    params = lora_params(model)
    R = len(rts)
    # 第 1 遍:只算每条 rollout 的总梯度 u_i(CPU fp16,r=32 时 ≈9 GB)
    U_rows = []
    for i, rt in enumerate(rts):
        logps = token_logps(model, rt["ids"], rt["plen"], device)
        u = torch.autograd.grad(logps.sum(), params)
        U_rows.append(flat_grads(u).half().cpu())
        del logps, u
        if (i + 1) % log_every == 0:
            print(f"[d4-lora] pass1 rollout {i + 1}/{R}")
            torch.cuda.empty_cache()
    U_gpu = torch.stack(U_rows).to(device).float()  # [R, D] fp32(r=32 ≈18 GB, r=16 ≈9 GB)
    del U_rows
    gc.collect()
    # 第 2 遍:逐个抽样 token 现算 g_t,立刻点积,不缓存(峰值 = U + 单条计算图)
    K = torch.zeros(R, n_samp, dtype=torch.float32)
    done = 0
    for ri, lst in samp_by_roll.items():
        logps = token_logps(model, rts[ri]["ids"], rts[ri]["plen"], device)
        for (si, pos) in lst:
            g = flat_grads(torch.autograd.grad(logps[pos], params,
                                               retain_graph=True))
            K[:, si] = (U_gpu @ g).cpu()
            del g
        done += len(lst)
        del logps
        if done % 256 < len(lst):
            print(f"[d4-lora] pass2 token {done}/{n_samp}")
            torch.cuda.empty_cache()
    del U_gpu
    gc.collect()
    torch.cuda.empty_cache()
    return K


# --------------------------------------------------------- lm_head 侧核矩阵
@torch.no_grad()
def lm_kernel_for_question(model, rts, samp_rows, device):
    """闭式核:K_lm[i,t] = Σ_{s∈i} ⟨h_s,h_t⟩[𝟙{y_s=y_t} − π_s(y_t) − π_t(y_s) + ⟨π_s,π_t⟩]。
    samp_rows: [(rollout, pos), ...],顺序即样本序号。两遍流式。"""
    R, n = len(rts), len(samp_rows)
    want = {}
    for si, (ri, pos) in enumerate(samp_rows):
        want.setdefault(ri, []).append((si, pos))

    def forward_one(rt):
        ids = rt["ids"].to(device).unsqueeze(0)
        out = model(input_ids=ids, output_hidden_states=True)
        lg = out.logits[0, rt["plen"] - 1:-1, :].float()
        h = out.hidden_states[-1][0, rt["plen"] - 1:-1, :].float()
        y = ids[0, rt["plen"]:]
        return lg, h, y

    hs, ps, ys = [None] * n, [None] * n, [None] * n
    for ri, lst in want.items():
        lg, h, y = forward_one(rts[ri])
        pr = F.softmax(lg, dim=-1)
        for (si, pos) in lst:
            hs[si] = h[pos].clone()
            ps[si] = pr[pos].clone()
            ys[si] = int(y[pos])
        del lg, h, pr
    H = torch.stack(hs)                            # [n, h] fp32
    P = torch.stack(ps)                            # [n, V] fp32(n=2000 ≈ 1.2 GB)
    Y = torch.tensor(ys, device=device)
    K = torch.zeros(R, n, dtype=torch.float32)
    for i, rt in enumerate(rts):
        lg, h, y = forward_one(rt)
        pr = F.softmax(lg, dim=-1)                 # [T, V] fp32
        C = h @ H.t()                              # ⟨h_s, h_t⟩        [T, n]
        M1 = pr[:, Y]                              # π_s(y_t)          [T, n]
        PT = P[:, y].t()                           # π_t(y_s)          [T, n]
        PP = pr @ P.t().float()                    # ⟨π_s, π_t⟩        [T, n]
        EQ = (y.unsqueeze(1) == Y.unsqueeze(0)).float()
        K[i] = (C * (EQ - M1 - PT + PP)).sum(0).cpu()
        del lg, h, pr, C, M1, PT, PP, EQ
        if (i + 1) % 64 == 0:
            print(f"[d4-lm] rollout {i + 1}/{R}")
            torch.cuda.empty_cache()
    del H, P
    torch.cuda.empty_cache()
    return K


# ------------------------------------------------------------- K 空间分析
def refs_and_stability(K, verd, samp_roll):
    """参考(全体+留一)与奇偶半参考。优势约定与主探针一致:
    参考用全体均值基线;半参考沿用全体优势、只累加本半(stages.py 同款)。"""
    R, n = K.shape
    adv = verd - verd.mean()
    idx = np.arange(n)
    own = K[samp_roll, idx]
    ref = K.T @ adv - adv[samp_roll] * own
    halves = {}
    half = np.arange(R) % 2
    for hv in (0, 1):
        a = np.where(half == hv, adv, 0.0)
        own_h = own * (half[samp_roll] == hv)
        halves[hv] = K.T @ a - a[samp_roll] * own_h
    return ref, halves


def wagree(a, b, w):
    m = (np.sign(a) != 0) & (np.sign(b) != 0)
    if m.sum() == 0:
        return float("nan")
    return float((w[m] * (np.sign(a[m]) == np.sign(b[m]))).sum() / w[m].sum())


def _acc_sums(est, ref, w):
    m = (np.sign(est) != 0) & (np.sign(ref) != 0)
    if m.sum() == 0:
        return 0.0, 0.0
    return float((w[m] * (np.sign(est[m]) == np.sign(ref[m]))).sum()), float(w[m].sum())


def analyze_question(K_lora, K_lm, verd, samp_roll, w_mass, stored_ref, rng,
                     jury_sizes, juries_per_size):
    R, n = K_lora.shape
    idx = np.arange(n)
    ref_lora, hv_lora = refs_and_stability(K_lora, verd, samp_roll)
    ref_lm, hv_lm = refs_and_stability(K_lm, verd, samp_roll)
    ones = np.ones(n)
    out = {
        "selfcheck_lmK_vs_stored": wagree(ref_lm, stored_ref, w_mass),
        "selfcheck_lmK_vs_stored_u": wagree(ref_lm, stored_ref, ones),
        "ref_agree_lm_lora_w": wagree(ref_lm, ref_lora, w_mass),
        "ref_agree_lm_lora_u": wagree(ref_lm, ref_lora, ones),
        "stability_lm_w": wagree(hv_lm[0], hv_lm[1], w_mass),
        "stability_lora_w": wagree(hv_lora[0], hv_lora[1], w_mass),
    }
    juries = {}
    for m in jury_sizes:
        sums = {k: [0.0, 0.0] for k in
                ("lora_own", "lm_own", "lora_vs_lm_ref", "lm_vs_lora_ref")}
        n_deg = 0
        for _ in range(juries_per_size):
            mem = rng.choice(R, size=m, replace=False)
            if verd[mem].min() == verd[mem].max():
                n_deg += 1
                continue
            a = np.zeros(R)
            a[mem] = verd[mem] - verd[mem].mean()   # 组内均值基线(训练约定)
            member = np.isin(samp_roll, mem)
            est = {}
            for name, K in (("lora", K_lora), ("lm", K_lm)):
                e = K.T @ a
                e = e - np.where(member, a[samp_roll] * K[samp_roll, idx], 0.0)
                est[name] = e
            for key, e, r in (("lora_own", est["lora"], ref_lora),
                              ("lm_own", est["lm"], ref_lm),
                              ("lora_vs_lm_ref", est["lora"], ref_lm),
                              ("lm_vs_lora_ref", est["lm"], ref_lora)):
                s, t = _acc_sums(e, r, w_mass)
                sums[key][0] += s
                sums[key][1] += t
        juries[str(m)] = {k: (v[0] / v[1] if v[1] else float("nan"))
                          for k, v in sums.items()}
        juries[str(m)]["degenerate"] = n_deg / juries_per_size
    out["juries"] = juries
    return out, ref_lora, ref_lm


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-token-sample", type=int, default=2000)
    ap.add_argument("--jury-sizes", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--juries-per-size", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_token_sample = 200
        args.jury_sizes = [4, 8]
        args.juries_per_size = 4

    device = "cuda"
    cfg = cfg_from_manifest(args.run_dir)
    if args.adapter:
        cfg.adapter_path = args.adapter
    out_dir = args.out or os.path.join(args.run_dir, "d4")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, "d4_manifest.json"), "w"),
              indent=1)
    qids = [m["idx"] for m in json.load(open(os.path.join(args.run_dir,
                                                          "questions.json")))]
    tok, model = load_student_with_lora(cfg, args.lora_r, args.lora_alpha,
                                        args.seed, device)
    summary = {"per_question": {}, "config": vars(args)}
    for qi in qids:
        qout = os.path.join(out_dir, f"q{qi}.npz")
        if os.path.exists(qout):
            print(f"[d4] q{qi} 已存在,复用其结果")
            d = np.load(qout, allow_pickle=True)
            summary["per_question"][str(qi)] = json.loads(str(d["res_json"]))
            continue
        meta, rolls = load_question(cfg, qi)
        rts = rollout_tensors(meta, rolls)
        verd = np.array([r["verdict"] for r in rolls], np.float64)
        tk = np.load(os.path.join(args.run_dir, f"q{qi}", "tokens.npz"))
        n_all = len(tk["pos"])
        rng = np.random.default_rng(args.seed * 7000 + qi)
        pick = np.sort(rng.choice(n_all, size=min(args.n_token_sample, n_all),
                                  replace=False))
        samp_roll = tk["rollout"][pick].astype(int)
        samp_pos = tk["pos"][pick].astype(int)
        w_mass = np.abs(tk["d"][pick] * tk["ref_iota"][pick]) + 1e-12
        stored_ref = tk["ref_iota"][pick].astype(np.float64)
        samp_rows = list(zip(samp_roll.tolist(), samp_pos.tolist()))
        samp_by_roll = {}
        for si, (ri, pos) in enumerate(samp_rows):
            samp_by_roll.setdefault(ri, []).append((si, pos))
        print(f"[d4] q{qi}: R={len(rts)}, 抽样 token {len(pick)}"
              f"(覆盖 {len(samp_by_roll)} 条 rollout)")

        K_lora = lora_kernel_for_question(model, rts, samp_by_roll,
                                          len(pick), device)
        with model.disable_adapter():
            K_lm = lm_kernel_for_question(model, rts, samp_rows, device)
        K_lora = K_lora.numpy().astype(np.float64)
        K_lm = K_lm.numpy().astype(np.float64)
        res, ref_lora, ref_lm = analyze_question(
            K_lora, K_lm, verd, samp_roll, w_mass, stored_ref, rng,
            args.jury_sizes, args.juries_per_size)
        np.savez_compressed(
            qout, K_lora=K_lora.astype(np.float32),
            K_lm=K_lm.astype(np.float32), verd=verd,
            samp_roll=samp_roll, samp_pos=samp_pos, w_mass=w_mass,
            ref_lora=ref_lora, ref_lm=ref_lm, ttype=tk["ttype"][pick],
            res_json=json.dumps(res))
        summary["per_question"][str(qi)] = res
        print(f"[d4] q{qi} 自检(lm-K vs 已存参考,质量加权): "
              f"{res['selfcheck_lmK_vs_stored']:.4f} | 跨子空间参考同号(质量加权): "
              f"{res['ref_agree_lm_lora_w']:.4f}")
        gc.collect()
        torch.cuda.empty_cache()

    _write_report(out_dir, summary, args)
    json.dump(summary, open(os.path.join(out_dir, "d4_summary.json"), "w"),
              indent=1, default=float)
    print("[d4] 完成 →", out_dir)


def _pool(summary, key, jkey=None, jfield=None):
    vals = []
    for q in summary["per_question"].values():
        if jkey:
            v = q.get("juries", {}).get(jkey, {}).get(jfield, float("nan"))
        else:
            v = q.get(key, float("nan"))          # 容错:旧版结果缺键时跳过,报告绝不崩
        if v == v:
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def _write_report(out_dir, summary, args):
    P = lambda x: "—" if x != x else f"{100 * x:.1f}%"
    L = ["# D4 —— 投影子空间对照(lm_head vs LoRA 块)\n"]
    L.append(f"配置:`{json.dumps(vars(args), ensure_ascii=False)}`\n")
    L.append("## 0. 运行时自检\n")
    L.append(f"- lm 侧 K-空间参考 vs 已存 tokens.npz 参考:质量加权同号率 "
             f"**{P(_pool(summary, 'selfcheck_lmK_vs_stored'))}**(应 ≈100%,验证整条 K 管线;"
             f"不加权 {P(_pool(summary, 'selfcheck_lmK_vs_stored_u'))},|ι|≈0 的零质量 token 受浮点精度影响可略低)。\n")
    L.append("## 1. 参考级:两个子空间在测同一个东西吗?\n")
    L.append(f"- 跨子空间参考同号率:质量加权 **{P(_pool(summary, 'ref_agree_lm_lora_w'))}**"
             f" · 不加权 {P(_pool(summary, 'ref_agree_lm_lora_u'))}")
    L.append(f"- 参考自稳定性(128/128,质量加权):lm_head "
             f"{P(_pool(summary, 'stability_lm_w'))} · LoRA "
             f"{P(_pool(summary, 'stability_lora_w'))}\n")
    L.append("## 2. 陪审团扫描(两子空间共用同一批陪审团,逐 token 配对,质量加权)\n")
    L.append("| m | LoRA→自参考 | lm→自参考 | LoRA→lm参考 | lm→LoRA参考 | 退化率 |")
    L.append("|---|---|---|---|---|---|")
    for m in args.jury_sizes:
        row = [P(_pool(summary, None, str(m), f)) for f in
               ("lora_own", "lm_own", "lora_vs_lm_ref", "lm_vs_lora_ref")]
        L.append(f"| {m} | {row[0]} | {row[1]} | {row[2]} | {row[3]} | "
                 f"{P(_pool(summary, None, str(m), 'degenerate'))} |")
    L.append("\n## 3. 预写判读\n")
    L.append("- 若 §1 跨子空间参考同号率 < ~80%:lm_head 测的量与 LoRA 训练相关的量存在实质分歧"
             "——换投影的动机独立于采样噪声;")
    L.append("- 若 §2 中『LoRA→自参考』明显高于『lm→自参考』:LoRA 子空间信噪比更好,"
             "定符号应搬进被训练的子空间;")
    L.append("- 若两者接近且 §1 同号率高:投影不是主要瓶颈,坚持大陪审团路线(计划 E5)。")
    L.append("\n*注:本次 LoRA 为全新初始化(B=0),前向与基座完全一致(缓存 rollout 依然有效),"
             "梯度=训练第 0 步真实所见;加 `--adapter` 指向训练中期检查点可得训练中的版本。*")
    open(os.path.join(out_dir, "d4_report.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()
