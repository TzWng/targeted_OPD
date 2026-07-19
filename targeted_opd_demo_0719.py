# @title new targeted
"""
targeted_opd_demo_v2_15b.py
=======================
Unified single-GPU demo comparing six on-policy update rules on GSM8K,
built around the Targeted-OPD paper. All six share ONE on-policy loop and
differ ONLY in the per-token reward r_t plugged into the KL-leashed
objective (paper eq 13) -- this is exactly the paper's Table 1 family,
plus faithful RLSD and G-OPD/ExOPD as external baselines.

Modes (each cross-checked against its source):
  - "opd"         : r_t = d_t                     paper Table 1 row 1; matches
                    Thinking Machines' tinker-cookbook OPD (advantage =
                    -reverse_KL = logP_T - logP_S, REINFORCE-style, discount 0).
  - "grpo"        : r_t = A_i = V_i - mean(V)     paper eq 6 verifier PG. Mean
                    baseline, no /sigma (paper's g_Q; equals Dr. GRPO / TRL
                    scale_rewards="none"). With ONE gradient step per batch of
                    rollouts the PPO ratio is exactly 1, so the ratio-free
                    REINFORCE form here is gradient-identical to GRPO.
  - "composition" : r_t = A_i * d_t               paper Table 1 row 3, the
                    verifier-signed composition -- the baseline Corollary 1
                    dominates. REQUIRED for testing the paper's claim.
  - "rlsd"        : faithful Self-Distilled RLVR (arXiv:2604.03128):
                    A_z = (V - mu)/(sigma + 1e-4)  (z-scored, their eq)
                    w_t = exp(sign(A_z) * d_t)     (their eq 14-15)
                    r_t = min(w_t*A_z, clip(w_t, 1-eps, 1+eps)*A_z)  (eq 16
                    pessimistic min-clip), eps = 0.2 (their value).
                    NOTE: in the RLSD paper the "teacher" is the SAME model
                    conditioned on the reference answer (privileged
                    self-teacher). Here we use the size-gap teacher for all
                    teacher-based modes, as a controlled comparison -- a
                    deliberate adaptation, say so when reporting.
  - "gopd"        : G-OPD / ExOPD (arXiv:2602.12125, RUC + Tencent Hunyuan):
                    r_t = d_t + (lambda-1) * (logP_T - logP_ref)
                    their eq 11/14 rewritten in ascent form; lambda=1 is
                    exactly OPD, lambda>1 extrapolates past the teacher
                    (ExOPD; their swept choice 1.25). pi_ref is the FROZEN
                    INITIAL student -- their strong-to-weak default is the
                    student base model, which is exactly what disabling the
                    LoRA adapters gives (LoRA init is the identity), so no
                    third model is loaded. NOTE: pi_ref must NOT be the
                    live student -- with pi_ref = pi_theta the reward
                    collapses to lambda*d_t, i.e. OPD times a constant.
  - "targeted"    : r_t = sign(iota_t) * |d_t|    paper Algorithm 1 / eq 12
                    (RL view): REINFORCE loss, signs only the SAMPLED token.
                    Influence via the last-layer rank-one identity (eq 15),
                    with EXACT leave-one-out gradients g_Q^(-i) (Appendix B)
                    and a neutral bin (Algorithm 1 line 4).
  - "targeted_kl" : paper Algorithm 2 (divergence view) -- THE PAPER'S
                    PRIMARY ALGORITHM. Per visited state, whole-vocabulary
                    influence iota(y) = (G h_t)_y - E_pi[G h_t] (eq 15, with
                    exact per-rollout LOO), tilted target
                        mu_t = softmax(log pi_theta + sign(iota(.))*|d(.)|),
                    d clipped to |d| <= tilt_clip (Algo 2 line 4), and the
                    closed-form loss sum_t KL(pi_theta(.|s_t) || sg(mu_t)).
                    No REINFORCE: the gradient carries no sampling variance;
                    only the sign of iota stays noisy. Also injects mass on
                    UNSAMPLED tokens -- the capability-transfer channel
                    Algorithm 1 lacks. Memory: needs the teacher's full
                    logits [B,L,V]; use prompts_per_step=1 below 40GB.

Design decisions locked to the paper:
  * reference-free: pi_ref = pi_theta  =>  d_t = logP_T - logP_S.detach()
    (paper section 3); the KL-leash gradient vanishes at the current point
    (section 2.2), so kl_coef = 0 is the paper-consistent default.
  * g_Q uses the mean baseline A_i = V_i - p_theta(x) with p_theta estimated
    per question from the group (paper eq 6 / Algorithm 1 line 1). No /sigma.
  * The influence gradient G is formed ONCE PER BATCH across all prompts
    (paper section 4: "formed once per batch by averaging ... over the
    sampled rollouts"). G here is an unnormalized sum; only sign(iota) is
    used, so the 1/N scale is irrelevant (LOO subtraction uses the same
    sum convention, so it stays consistent).
  * EXACT leave-one-out: token t of rollout i is scored against G - G_i
    (drop the WHOLE rollout, Algorithm 1 line 2), not just the self-token
    term. Implemented via per-rollout Gram matrices, one [T,V]@[V,T]
    matmul per rollout -- cheap. The self-token diagonal of the correction
    equals Appendix B's bias term, which is a useful sanity check.
  * Neutral bin (Algorithm 1 line 4): tokens whose |iota| falls in the
    bottom `neutral_q` quantile get r_t = 0. The paper does not fix a
    threshold; 0.05 is our operationalization, set 0.0 to disable.

Model choice (from the source papers' small-scale setups):
  * Student  Qwen/Qwen2.5-0.5B-Instruct      GSM8K pass@1 ~49.6 -- inside the
    15-60% band needed so groups of G rollouts mix correct/wrong.
  * Teacher  Qwen/Qwen2.5-Math-1.5B-Instruct GSM8K ~84.8, ~3GB fp16,
    inference-only; same Qwen2.5 tokenizer/vocab (151936). 4K context is
    plenty here (prompt<=384 + 256 new tokens).
    Alternative teacher: Qwen/Qwen2.5-1.5B-Instruct (73.2, no Math-format
    caveats, smaller gap).
  * Group size G=8 matches RLSD / HERO / RLAD / SG-OPD / CAST; temperature
    1.0 matches them too AND makes the sampling policy identical to the
    policy being optimized (no temperature mismatch in logP).
  * LoRA precedent: CAST (r=64) and Thinking Machines (r=32-128) both train
    OPD-style with LoRA; GKD (77M student <- 3B teacher on GSM8K) and
    MiniLLM (GPT-2 124M <- 1.5B) validate small students.

Memory design (after the chunked rework): NO full-vocab fp32 tensor is ever
materialized in the RL-view path. logP_S uses a row-chunked logsumexp with a
custom backward; influence softmaxes fp32 per chunk; teacher/ref forwards run
4 rows at a time. Peak @ B=16, L=1024 is ~23GB (was ~40GB+):
  student logits bf16 5GB (graph) + backward grad 5GB + models 7GB
  + vLLM@0.10 4GB + chunk transients ~1.5GB.
Tiers: T4 16GB 1x6 fp16 | L4 24GB 1x8 | A100 40GB 2x8 (default).
targeted_kl still needs full teacher logits: keep prompts_per_step=1 there.
Start tiny, confirm it steps, then scale.

Data (v3):
  * train zwhe99/DeepMath-103K (question / final_answer latex / difficulty
    float ~3-9): filtered to [dm_diff_min, dm_diff_max] -- tune via preflight
    so the student's pass@1 lands in the 5-25% band.
  * eval HuggingFaceH4/MATH-500 (problem / answer latex / level), greedy,
    \boxed{} extraction; eval_n=200 during training, full 500 for finals.
  * dataset="gsm8k" restores the Experiment-1 setup (#### verifier).
Verifier: last \boxed{...} (nested-brace parser) -> math-verify when
installed (community standard; STRONGLY recommended, gold is latex) ->
normalized-string/float fallback.
vLLM eval (use_vllm_eval=True): base weights load once into a vLLM engine
(gpu_memory_utilization=vllm_gpu_frac); the LoRA adapter is saved (~35MB)
and hot-swapped per eval via LoRARequest. ~5-10x faster than batched HF.

Install (Colab, ONE line -- vllm pins its own torch, so install everything
together in a FRESH runtime and restart once if prompted):
    !pip install -U vllm math-verify "transformers>=4.44" peft datasets accelerate
"""

import re
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


# ----------------------------------------------------------------------------- #
# Config
# ----------------------------------------------------------------------------- #
@dataclass
class Config:
    mode: str = "targeted"        # opd | grpo | composition | rlsd | gopd | targeted | targeted_kl
    # LOCAL-FIRST loading (2026-07-14): model weights and datasets live under
    # Drive's OPD folder (manually downloaded from the HF website), so no HF
    # network call is needed. load_models() copies the model dirs to the VM's
    # local disk once per session (Drive FUSE is slow to read); load_data()
    # falls back to the hub only if a local folder is missing.
    # 1.5B-STUDENT FILE (2026-07-17): capacity-parity OPD replication that
    # satisfies the tinker/G-OPD conditions. student = Qwen2.5-Math-1.5B
    # (math base) FULL-FT on GSM8K to align chat format WITHOUT closing the gap
    # (MATH-500 0.406 vs teacher 0.648 -> 24pt headroom). SFT-v2 is TIED
    # (full fine-tune trained lm_head so it SELF-STOPS at <|im_end|>, and tied
    # structure keeps vLLM fast). teacher = the math-Instruct sibling (same
    # base, +SFT/RM/GRPO -> in student's support). stop_token_ids kept as a
    # zero-cost safety net below.
    student: str = "/content/drive/MyDrive/OPD/Qwen2.5-Math-1.5B-SFT-v2"
    teacher: str = "/content/drive/MyDrive/OPD/Qwen2.5-Math-1.5B-Instruct"
    data_dir: str = "/content/drive/MyDrive/OPD"  # GSM8K/ DeepMath/ MATH500/
    dataset: str = "deepmath"          # deepmath (train DeepMath-103K, eval
                                       # MATH-500, \boxed verifier) | gsm8k
                                       # (Experiment-1 behavior, #### verifier)
    dm_diff_min: float = 0.0           # DeepMath difficulty band. Preflight
    dm_diff_max: float = 3.0           # (2026-07-10, 24q x 8 samples/band):
                                       # (0,3.0] n=3434 pass=0.188 mixed=0.50 <- pick
                                       # (3,3.5] n=990  pass=0.042 mixed=0.17
                                       # (3.5,4] n=5430 pass=0.109 mixed=0.38
                                       # (4,4.5] n=4391 pass=0.068 mixed=0.21

    # rollout / batch  (memory knobs -- shrink first if you OOM; see tiers above)
    prompts_per_step: int = 4          # questions per MICRO-batch (one forward)
    grad_accum: int = 4                # micro-batches accumulated per optimizer
                                       # step (TRL's gradient_accumulation_steps).
                                       # Effective update batch =
                                       # prompts_per_step * grad_accum * G
                                       # = 1*4*8 = 32 sequences, memory-free.
                                       # NOTE: influence/advantages stay
                                       # per-micro-batch (TRL semantics);
                                       # accumulation smooths the gradient only.
    group_size: int = 4                # rollouts per question (G) -- papers use 8
    max_new_tokens: int = 512          # MATH solutions run long; 384 truncates
                                       # too many rollouts -> corrupted verdicts
    max_prompt_tokens: int = 512       # MATH problems reach ~450 tokens; 384
                                       # would truncate the longest ones
    temperature: float = 1.0           # 1.0: sampling policy == optimized policy
    top_p: float = 1.0
    presence_penalty: float = 0.0      # Qwen3 non-thinking rec: 1.5 to kill
                                       # repetition loops (greedy/temp1 both loop
                                       # on hard math). 0 keeps Qwen2.5 unchanged.
    enable_thinking: bool = False      # Qwen3 chat template: suppress <think>
                                       # blocks. Ignored by the Qwen2.5 template.
    rollout_is_correction: bool = False  # truncated importance-sampling (TIS)
                                       # correction for the vLLM(sample) vs HF
                                       # (score) mismatch -- rho=pi_train/pi_roll,
                                       # clamp high, drop extreme-low, detached
                                       # onto r_t. (G-OPD's rollout correction.)
    rollout_is_threshold: float = 2.0  # clamp IS weight to <=thr; drop tokens
                                       # with rho < 1/thr as off-policy outliers
    mask_eos_in_loss: bool = True      # exclude the <|im_end|> stop token from the
                                       # OPD reward/loss so training NEVER erodes the
                                       # SFT-learned stopping (d_t on im_end can be
                                       # negative when the teacher is more verbose).
                                       # Stopping stays frozen at the SFT level.

    # checkpointing (LoRA adapter + optimizer + step/ptr/eval history)
    run_prefix: str = "s15b"           # ckpt dir prefix: {run_prefix}_{tag}_{dataset}
                                       # _seed{seed}. Encodes the model/setup so runs
                                       # with different bases NEVER share a dir (set
                                       # e.g. "q3-1.7b" for the Qwen3-1.7B-Base-SFT run
                                       # to not collide with old s15b_ checkpoints).
    save_dir: Optional[str] = "/content/drive/MyDrive/OPD"
    save_every: int = 100              # also saves at final; None save_dir = off
    resume: bool = False               # OFF by default (2026-07-14): resuming
                                       # across variants that shared a ckpt dir
                                       # silently mixed runs once. Only turn on
                                       # to continue the SAME variant in the
                                       # SAME dir after an interruption.

    # optimization
    lr: float = 1e-5
    steps: int = 300
    warmup_steps: int = 20             # linear warmup 0->lr, then CONSTANT lr
                                       # (never decays, per directive). Set 0 for
                                       # a fully flat lr from step 1.
    lr_min_ratio: float = 0.1          # UNUSED now (no cosine decay). Kept for
                                       # config compatibility.
    kl_coef: float = 0.0               # k3 KL leash. DEFAULT 0 to match the
                                       # G-OPD paper's harness (their App. B
                                       # Tables 4-6: GRPO KL coef = 0.0, no
                                       # extra KL in distillation) and modern
                                       # practice (DAPO/LUFFY). DeepSeekMath's
                                       # classic beta=0.04 available as an
                                       # ablation; >0 costs one extra no-grad
                                       # ref forward per step.
    kl_ref: str = "init"               # "init" = frozen initial policy (GRPO
                                       # canonical anchor) | "teacher" (OPD-
                                       # flavored leash). targeted_kl mode is
                                       # unaffected (its loss IS a KL already).
    weight_decay: float = 0.01         # previously AdamW's silent default;
                                       # now explicit (HERO uses 0.1, unsloth
                                       # 0.001 -- 0.01 is the middle ground)
    grad_clip: float = 1.0
    seed: int = 0

    # rlsd (faithful to arXiv:2604.03128)
    rlsd_eps: float = 0.2              # their eps_w

    # gopd (faithful to arXiv:2602.12125)
    gopd_lambda: float = 1.25          # their ExOPD value; 1.0 recovers OPD

    # targeted / influence
    influence_chunk: int = 256         # states per chunk when forming G / G h_t
    leave_one_out: bool = True         # exact per-rollout LOO (Algorithm 1 line 2)
    amp_same: float = 1.25             # line-4 coefficient on confirmed-ALIGNED
                                       # tokens: amplified follow (= gopd_lambda,
                                       # so the gopd comparison shares one
                                       # budget). {amp_same,1,-amp_flip} =
                                       # {lambda,1,1-lambda} at lambda=1.25.
    amp_flip: float = 0.25             # magnitude (in |d_t| units) on confirmed-
                                       # CONTRADICTED tokens; applied NEGATED
                                       # (r=-amp_flip*d_t = quarter-strength
                                       # step in the iota direction). 1.0 with
                                       # amp_same=1.0 recovers the symmetric
                                       # flip variant. Coefficients are encoded
                                       # into the ckpt dir name -- sweeps
                                       # auto-isolate.
    neutral_q: float = 0.7             # UNUSED since the split-half gate
                                       # replaced the |iota|-quantile test
                                       # (quantile was mass-confounded); kept
                                       # only for config compatibility.
    tilt_clip: float = 5.0             # Algo 2 line 4: clip |d(y)| <= c (anti-runaway)

    # eval
    eval_every: int = 50
    rollout_print_every: int = 0       # >0: print 1 TRAINING rollout every N steps
                                       # (near-free -- reuses a rollout already
                                       # generated this step). Cheap way to watch
                                       # output quality without the slow full eval.
    eval_show_samples: int = 3         # print this many full responses each eval
                                       # (fixed first-N questions, so you can watch
                                       # the SAME items across steps -- catches
                                       # stop-token drift / rambling during training)
    eval_n: int = 500                  # full MATH-500 during training (vLLM eval
                                       # is fast now that vllm_gpu_frac=0.35)
    eval_batch: int = 32               # questions per generate call (left-padded);
                                       # 0.5B KV cache is tiny, 32-64 is safe
    eval_max_new_tokens: int = 1024    # eval-only generation cap; 640 truncated
                                       # 36% of MATH answers for the 0.5B student
    use_vllm_eval: bool = True         # vLLM engine + LoRA hot-swap for eval;
                                       # auto-falls back to HF if vllm missing
    use_vllm_rollouts: bool = True     # vLLM also generates TRAINING rollouts
                                       # (adapter re-saved each step, ~1-2s;
                                       # cuts ~45s of HF generate per step).
                                       # Training math unchanged: logP/hidden
                                       # still come from the HF forward.
    vllm_gpu_frac: float = 0.1        # 1.5B: 0.10 starves the KV cache and
                                       # rollout generation crawls (co-located
                                       # HF training still leaves ~40GB on an
                                       # 80GB card). 0.10 was fine for 0.5B/
                                       # short seqs; 1.5B needs ~28GB for vLLM
                                       # or rollouts run 10-30x slower.
                                       # (0.5B weights + eval KV fit in ~4GB)

    attn_impl: str = "sdpa"            # sdpa is ~20-40% faster than eager for
                                       # generation; hidden_states still exact
    dtype: Optional[torch.dtype] = None  # None: bf16 if supported else fp16 (T4!)
    device: str = "cuda"


# ----------------------------------------------------------------------------- #
# Data + rule verifier (GSM8K: gold answer after "####")
# ----------------------------------------------------------------------------- #
SYS_GSM8K = ("You are a helpful math assistant. Solve the problem step by step, "
             "then give the final answer on a new line as '#### <number>'.")
SYS_MATH = ("Please reason step by step, and put your final answer within "
            "\\boxed{}.")   # Qwen2.5-Math's native answer format

_num = re.compile(r"-?\d[\d,]*\.?\d*")

# math-verify (community-standard latex-equivalence checker, used by
# LUFFY/verl) is strongly recommended: gold answers on MATH/DeepMath are
# latex expressions. Falls back to normalized string / float comparison.
try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    _HAVE_MV = True
except Exception:
    _HAVE_MV = False

def extract_boxed(text: str):
    # content of the LAST \boxed{...}, nested-brace aware
    i = text.rfind("\\boxed")
    if i == -1:
        return None
    j = text.find("{", i)
    if j == -1:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None   # unbalanced (truncated generation)

def extract_pred(text: str) -> str:
    b = extract_boxed(text)
    if b is not None:
        return b.strip()
    if "####" in text:
        tail = text.split("####")[-1]
        m = _num.search(tail)
        if m:
            return m.group(0).replace(",", "")
    nums = _num.findall(text)                 # last-number fallback
    return nums[-1].replace(",", "") if nums else ""

def _norm(s: str) -> str:
    s = s.strip().strip("$").replace(" ", "").replace(",", "")
    for t in ("\\left", "\\right", "\\!", "\\,"):
        s = s.replace(t, "")
    return s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")

def verify(pred: str, gold: str) -> float:
    if not pred:
        return 0.0
    if _HAVE_MV:
        try:
            return 1.0 if _mv_verify(_mv_parse(gold), _mv_parse(pred)) else 0.0
        except Exception:
            pass
    p, g = _norm(pred), _norm(gold)
    if p == g:
        return 1.0
    try:
        return 1.0 if abs(float(p) - float(g)) < 1e-4 else 0.0
    except ValueError:
        return 0.0

def build_prompt(cfg, tok, question: str) -> str:
    sys = SYS_GSM8K if cfg.dataset == "gsm8k" else SYS_MATH
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": question}]
    # enable_thinking is a Qwen3 template kwarg; Qwen2.5's template doesn't take
    # it, so fall back cleanly on TypeError.
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=cfg.enable_thinking)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def gold_answer(sol: str) -> str:
    # GSM8K solutions end with "#### <int>"
    return sol.split("####")[-1].strip().replace(",", "")

def load_data(cfg):
    """Return (train_items, test_items); each item is {'question', 'gold'}."""
    if cfg.dataset == "gsm8k":
        import os
        loc = os.path.join(cfg.data_dir, "GSM8K")
        if os.path.isdir(loc):
            # local parquet copy (downloaded from the HF website) -- same
            # columns (question/answer) as the hub dataset, zero network
            ds = load_dataset("parquet", data_files={
                "train": os.path.join(loc, "train-00000-of-00001.parquet"),
                "test": os.path.join(loc, "test-00000-of-00001.parquet")})
        else:
            ds = load_dataset("openai/gsm8k", "main")
        train = [{"question": ex["question"], "gold": gold_answer(ex["answer"])}
                 for ex in ds["train"].shuffle(seed=cfg.seed)]
        test = [{"question": ex["question"], "gold": gold_answer(ex["answer"])}
                for ex in ds["test"].select(range(cfg.eval_n))]
    elif cfg.dataset == "deepmath":
        import os
        dm_loc = os.path.join(cfg.data_dir, "deepmath")  # Drive folder is lowercase
        if os.path.isdir(dm_loc):
            # local parquet shards (data/train-0000x-of-00010.parquet from
            # the HF website) -- same columns, zero network
            tr = load_dataset("parquet",
                              data_files=os.path.join(dm_loc, "**/*.parquet"),
                              split="train")
        else:
            tr = load_dataset("zwhe99/DeepMath-103K", split="train")
        tr = tr.filter(lambda ex:
                       cfg.dm_diff_min <= ex["difficulty"] <= cfg.dm_diff_max)
        tr = tr.shuffle(seed=cfg.seed)
        print(f"DeepMath difficulty [{cfg.dm_diff_min}, {cfg.dm_diff_max}]: "
              f"{len(tr)} problems")
        train = [{"question": ex["question"], "gold": ex["final_answer"]}
                 for ex in tr]
        m5_loc = os.path.join(cfg.data_dir, "MATH500", "test.jsonl")
        if os.path.isfile(m5_loc):
            # local jsonl (downloaded from the HF website); the generic json
            # loader names its split "train" regardless of content
            te = load_dataset("json", data_files=m5_loc, split="train")
        else:
            te = load_dataset("HuggingFaceH4/MATH-500", split="test")
        test = [{"question": ex["problem"], "gold": ex["answer"]}
                for ex in te.select(range(min(cfg.eval_n, len(te))))]
    else:
        raise ValueError(f"unknown dataset {cfg.dataset}")
    return train, test


# ----------------------------------------------------------------------------- #
# Model loading
# ----------------------------------------------------------------------------- #
def load_models(cfg: Config):
    if cfg.dtype is None:
        cfg.dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                     else torch.float16)          # T4 has no bf16
    print(f"dtype: {cfg.dtype}")

    # If the model paths live on Drive, copy them to the VM's local disk once
    # per session: FUSE reads are slow, and vLLM's engine core also loads
    # weights from this path. Local paths / hub ids pass through untouched.
    import os
    import shutil
    for attr in ("student", "teacher"):
        p = getattr(cfg, attr)
        if p.startswith("/content/drive"):
            dst = os.path.join("/content/local_models", os.path.basename(p))
            if not os.path.isdir(dst):
                print(f"[local] copying {p} -> {dst} (one-time, ~1-3 min)")
                shutil.copytree(p, dst)
            setattr(cfg, attr, dst)

    tok = AutoTokenizer.from_pretrained(cfg.student)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Qwen2.5: pad <|endoftext|> != eos <|im_end|>, so the completion mask
    # below keeps the EOS token trainable, as it should be.
    tok.padding_side = "left"          # for batched generation

    student = AutoModelForCausalLM.from_pretrained(
        cfg.student, dtype=cfg.dtype, attn_implementation=cfg.attn_impl
    ).to(cfg.device)
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    student = get_peft_model(student, lora)
    student.print_trainable_parameters()

    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher, dtype=cfg.dtype, attn_implementation=cfg.attn_impl
    ).to(cfg.device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # sanity: shared vocab is required for token-aligned logprobs
    assert student.config.vocab_size == teacher.config.vocab_size, \
        "Teacher and student must share the tokenizer/vocab."
    return tok, student, teacher


# ----------------------------------------------------------------------------- #
# Rollout: sample G completions per prompt, return token ids + masks
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def sample_rollouts(cfg, tok, student, prompt_text: str):
    enc = tok(prompt_text, return_tensors="pt", truncation=True,
              max_length=cfg.max_prompt_tokens).to(cfg.device)
    plen = enc.input_ids.shape[1]
    student.eval()
    out = student.generate(
        **enc,
        do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p,
        max_new_tokens=cfg.max_new_tokens,
        num_return_sequences=cfg.group_size,
        pad_token_id=tok.pad_token_id,
    )
    student.train()
    # out: [G, plen + gen]
    full = out                                    # includes prompt
    comp = out[:, plen:]                          # completion ids
    texts = tok.batch_decode(comp, skip_special_tokens=True)
    return full, plen, texts


# ----------------------------------------------------------------------------- #
# Memory-light logsumexp over the vocab dim of [B, L, V] logits, one batch row
# at a time. Never materializes a full-vocab fp32 copy WITH GRAD (the old
# F.log_softmax(s_logits.float()) retained ~10GB at B=16, L=1024).
# Backward is exact: d/dlogits = softmax(logits) * grad_out, also per row.
# ----------------------------------------------------------------------------- #
class _RowChunkedLogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):                       # [B, L, V], bf16/fp16/fp32
        out = torch.empty(logits.shape[0], logits.shape[1],
                          device=logits.device, dtype=torch.float32)
        for i in range(logits.shape[0]):
            out[i] = torch.logsumexp(logits[i].float(), dim=-1)
        ctx.save_for_backward(logits, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):                    # grad_out [B, L] fp32
        logits, out = ctx.saved_tensors
        grad = torch.empty_like(logits)
        for i in range(logits.shape[0]):
            grad[i] = (torch.exp(logits[i].float() - out[i].unsqueeze(-1))
                       * grad_out[i].unsqueeze(-1)).to(logits.dtype)
        return grad


def _ref_logP(cfg, student, input_ids, attn, tgt, device):
    # Frozen initial policy (= base model, LoRA off) log-probs on the sampled
    # tokens; row-chunked like the teacher pass. Used by gopd and the KL leash.
    with torch.no_grad(), student.disable_adapter():
        out = torch.empty(input_ids.shape[0], tgt.shape[1],
                          device=device, dtype=torch.float32)
        for i in range(0, input_ids.shape[0], 4):
            rl = student(input_ids=input_ids[i:i + 4],
                         attention_mask=attn[i:i + 4]).logits[:, :-1, :]
            for j in range(rl.shape[0]):
                lp = F.log_softmax(rl[j].float(), dim=-1)
                out[i + j] = lp.gather(-1, tgt[i + j].unsqueeze(-1)).squeeze(-1)
            del rl
    return out


def _lr_at(cfg, step):
    # CONSTANT lr -- NEVER decays (user directive). Optional linear warmup only:
    # ramps 0 -> cfg.lr over warmup_steps, then holds cfg.lr flat forever.
    # Set warmup_steps=0 for a fully flat lr from step 1. lr_min_ratio is unused.
    if step <= cfg.warmup_steps:
        return cfg.lr * step / max(cfg.warmup_steps, 1)
    return cfg.lr


# ----------------------------------------------------------------------------- #
# One optimizer step
# ----------------------------------------------------------------------------- #
def train_step(cfg, tok, student, teacher, opt, batch_questions, batch_gold,
               zero_grad=True, opt_step=True, loss_scale=1.0):
    # zero_grad/opt_step/loss_scale implement gradient accumulation: the
    # caller zeroes on the first micro-batch, steps on the last, and scales
    # each micro-loss by 1/grad_accum. Reported metrics stay UNscaled.
    device = cfg.device
    # ---- collect rollouts across the micro-batch of prompts ----
    groups = None
    if cfg.use_vllm_rollouts:
        try:
            groups = _vllm_rollouts(
                cfg, tok, student,
                [build_prompt(cfg, tok, q) for q in batch_questions],
                reuse_adapter=not zero_grad)
        except ImportError:
            tqdm.write("[rollout] vllm not installed -- falling back to HF "
                       "generate for training rollouts (much slower).")
            cfg.use_vllm_rollouts = False

    # grab one rollout (first prompt, first sample) for cheap in-loop inspection
    _sample = None
    if groups and groups[0][1]:
        _c, _t = groups[0][1][0][0], groups[0][1][0][1]
        _sample = {"text": _t, "stop": tok.convert_tokens_to_ids("<|im_end|>") in _c,
                   "len": len(_c)}

    seqs, comp_masks, adv_mean, adv_z, roll_logps = [], [], [], [], []
    for gi, (q, g) in enumerate(zip(batch_questions, batch_gold)):
        if groups is not None:
            pids, rolls = groups[gi]
            plen = len(pids)
            texts = [r[1] for r in rolls]
            id_list = [torch.tensor(pids + r[0], device=device, dtype=torch.long)
                       for r in rolls]
            lp_list = [r[2] for r in rolls]        # rollout logp / completion token
        else:
            ptext = build_prompt(cfg, tok, q)
            full, plen, texts = sample_rollouts(cfg, tok, student, ptext)
            id_list = [full[i] for i in range(full.shape[0])]
            lp_list = [None] * len(id_list)        # HF fallback: no rollout logp
        Vs = torch.tensor([verify(extract_pred(t), g) for t in texts], device=device)
        A = Vs - Vs.mean()                        # paper eq 6 mean baseline, no /sigma
        Az = (Vs - Vs.mean()) / (Vs.std() + 1e-4) # RLSD's z-scored group advantage
        for i, ids in enumerate(id_list):
            mask = torch.zeros_like(ids, dtype=torch.bool)
            mask[plen:] = ids[plen:] != tok.pad_token_id
            rlp = torch.zeros(ids.shape[0], device=device)
            if lp_list[i] is not None:             # align rollout logp to completion
                lp = torch.tensor(lp_list[i], device=device, dtype=torch.float32)
                rlp[plen:plen + lp.shape[0]] = lp
            seqs.append(ids)
            comp_masks.append(mask)
            adv_mean.append(A[i])
            adv_z.append(Az[i])
            roll_logps.append(rlp)

    # pad to same length; rollout i is row i throughout
    L = max(s.shape[0] for s in seqs)
    input_ids = torch.full((len(seqs), L), tok.pad_token_id, device=device, dtype=torch.long)
    cmask = torch.zeros((len(seqs), L), dtype=torch.bool, device=device)
    rollout_lp = torch.zeros((len(seqs), L), device=device)
    for i, (s, m) in enumerate(zip(seqs, comp_masks)):
        input_ids[i, :s.shape[0]] = s
        cmask[i, :m.shape[0]] = m
        rollout_lp[i, :roll_logps[i].shape[0]] = roll_logps[i]
    attn = (input_ids != tok.pad_token_id).long()
    rollout_lp = rollout_lp[:, 1:]                 # [B, L-1], aligned with tgt
    A_vec = torch.stack(adv_mean)                 # [B]
    Az_vec = torch.stack(adv_z)                   # [B]

    # log fraction of non-degenerate groups (mixed correct/wrong) -- key signal
    nondegen = (A_vec.abs() > 1e-6).float().mean().item()

    # ROLLOUT length instrumentation: completion tokens per rollout and the
    # fraction that hit the training generation cap (== truncated verdicts)
    comp_len = cmask.sum(1)
    len_mean = comp_len.float().mean().item()
    cap_frac = (comp_len >= cfg.max_new_tokens).float().mean().item()

    # ---- student forward (grad) : logits + last hidden ----
    # hidden_states[-1] is post-final-norm, i.e. exactly the h_t with
    # logits = W h_t that eq 15 requires.
    student.train()
    s_out = student(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
    s_logits = s_out.logits[:, :-1, :]            # predict token t+1 at position t
    hidden = s_out.hidden_states[-1][:, :-1, :]   # [B, L-1, h]
    tgt = input_ids[:, 1:]                        # [B, L-1]
    tmask = cmask[:, 1:]                          # completion tokens only
    if cfg.mask_eos_in_loss:                      # freeze SFT-learned stopping:
        im_end = tok.convert_tokens_to_ids("<|im_end|>")   # drop the stop token from
        tmask = tmask & (tgt != im_end)           # ALL modes' loss/reward/influence

    # ---- Algorithm 2 (divergence view, the paper's primary algorithm) takes
    # its own path: closed-form KL to the tilted target, no REINFORCE ----
    if cfg.mode == "targeted_kl":
        with torch.no_grad():
            t_logits_full = teacher(input_ids=input_ids,
                                    attention_mask=attn).logits[:, :-1, :]
        loss, iota, d_samp = targeted_kl_loss(cfg, s_logits, t_logits_full,
                                              hidden.detach(), tgt, tmask, A_vec)
        if zero_grad:
            opt.zero_grad()
        (loss * loss_scale).backward()
        if opt_step:
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], cfg.grad_clip)
            opt.step()
        with torch.no_grad():
            di = (d_samp * iota)[tmask]
            denom = di.abs().sum().clamp(min=1e-8)
            probe = (di[di < 0].abs().sum() / denom).item() if di.numel() > 0 else float("nan")
        return {"loss": loss.item(), "pg": loss.item(),
                "reward_correct": A_vec.gt(0).float().mean().item(),
                "nondegen_frac": nondegen, "misaligned_mass": probe,
                "len_mean": len_mean, "cap_frac": cap_frac, "sample": _sample}

    del s_out   # drop the all-layer hidden_states tuple

    # logP_S = logit_y - logsumexp, WITHOUT materializing a full-vocab fp32
    # log_softmax with grad (was the single biggest allocation, ~10GB @ B=16)
    logZ = _RowChunkedLogSumExp.apply(s_logits)                         # [B, L-1] fp32
    logit_y = s_logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).float()
    logP_S = logit_y - logZ                                             # [B, L-1] (grad)

    # ---- teacher forward (no grad), chunked over rows: caps transient
    # logits at [4, L, V] bf16 (~1.2GB) instead of the full batch ----
    with torch.no_grad():
        logP_T = torch.empty(input_ids.shape[0], tgt.shape[1],
                             device=device, dtype=torch.float32)
        for i in range(0, input_ids.shape[0], 4):
            tl = teacher(input_ids=input_ids[i:i + 4],
                         attention_mask=attn[i:i + 4]).logits[:, :-1, :]
            for j in range(tl.shape[0]):
                lp = F.log_softmax(tl[j].float(), dim=-1)               # [L-1, V]
                logP_T[i + j] = lp.gather(-1, tgt[i + j].unsqueeze(-1)).squeeze(-1)
            del tl

    d_t = (logP_T - logP_S.detach())              # [B, L-1]  reference-free (paper sec 3)
    A_bt = A_vec.unsqueeze(1).expand_as(d_t)      # [B, L-1]

    # ---- reward per mode ----
    iota = None
    agree_rate = float("nan")
    if cfg.mode == "opd":
        r_t = d_t
    elif cfg.mode == "grpo":
        r_t = A_bt
    elif cfg.mode == "composition":
        r_t = A_bt * d_t                          # paper Table 1: verifier-signed comp.
    elif cfg.mode == "rlsd":
        Az_bt = Az_vec.unsqueeze(1).expand_as(d_t)
        w = torch.exp(Az_bt.sign() * d_t)         # RLSD eq 14-15
        r_t = torch.minimum(w * Az_bt,            # eq 16 pessimistic min-clip
                            w.clamp(1 - cfg.rlsd_eps, 1 + cfg.rlsd_eps) * Az_bt)
    elif cfg.mode == "gopd":
        # pi_ref = frozen initial student = base model (adapters off; exact
        # because LoRA initializes to the identity). One extra no-grad pass.
        logP_ref = _ref_logP(cfg, student, input_ids, attn, tgt, device)
        d_ref = logP_T - logP_ref                 # log(pi*/pi_ref), no grad
        r_t = d_t + (cfg.gopd_lambda - 1.0) * d_ref   # eq 11/14, ascent form
    elif cfg.mode == "targeted":
        iota, iota_alt = compute_influence(cfg, s_logits.detach(), hidden.detach(),
                                           tgt, tmask, A_vec)           # [B, L-1] x2
        sgn = iota.sign()
        # Algorithm 1 line 4 (AMENDED: split-half confidence + ASYMMETRIC
        # coefficients, 2026-07-14): a verdict requires the two INDEPENDENT
        # half-group influence-sign estimates to agree. Then
        #   confirmed-ALIGNED     : r = +amp_same * d_t   (amplified follow)
        #   confirmed-CONTRADICTED: r = -amp_flip * d_t   (attenuated reverse
        #                           = +amp_flip*sign(iota)|d_t|)
        #   uncertain             : r = d_t               (follow the teacher)
        # DAPO-style up/down asymmetry: the anchored direction (toward the
        # teacher) gets the big step, the unanchored reverse gets the small
        # one. GSM8K n=1319 paired eval: asym >= flip at 7/8 snapshots.
        agree = (sgn * iota_alt.sign()) > 0
        same = sgn == d_t.sign()
        r_t = torch.where(agree,
                          torch.where(same, cfg.amp_same * d_t,
                                      -cfg.amp_flip * d_t),
                          d_t)
        with torch.no_grad():                     # sign-quality meter
            valid = tmask & (iota != 0) & (iota_alt != 0)
            nv = valid.sum().item()
            agree_rate = (((sgn == iota_alt.sign()) & valid)
                          .sum().item() / nv) if nv > 0 else float("nan")
        # -- earlier variants, superseded: --
        # |iota|-quantile gate:  low = |iota| <= quantile(neutral_q); r=where(low,d,r)
        # paper-original line 4: neutral bin -> r_t = 0
    else:
        raise ValueError(cfg.mode)

    # ---- rollout importance-sampling (TIS) correction: sampling ran on vLLM
    # (bf16 / presence_penalty / stale adapter), scoring runs on the HF forward.
    # rho = pi_train / pi_rollout; clamp high (variance), drop extreme-low
    # outliers, detach, multiply onto r_t. Corrects the train/inference mismatch.
    is_reject = float("nan"); is_w_mean = float("nan")
    if cfg.rollout_is_correction and cfg.use_vllm_rollouts:
        with torch.no_grad():
            thr = cfg.rollout_is_threshold
            rho = torch.exp(logP_S.detach() - rollout_lp)     # pi_train / pi_rollout
            is_w = rho.clamp(max=thr)                         # truncate high weights
            keep = (rho >= 1.0 / thr).float()                # drop extreme-low outliers
            m = tmask
            is_reject = (((keep < 1) & m).sum() / m.sum().clamp(min=1)).item()
            is_w_mean = is_w[m].mean().item() if m.any() else float("nan")
        r_t = r_t * (is_w * keep)

    # ---- loss: -(r_t * logP_S) over completion tokens (+ optional KL leash) ----
    # Normalization is the global token mean for ALL modes (controlled
    # comparison). RLSD's own paper normalizes per-sequence (1/|y|) then over
    # the group; switch here if you want that exact weighting.
    r_t = r_t.detach()
    ntok = tmask.sum().clamp(min=1)
    pg_loss = -((r_t * logP_S) * tmask).sum() / ntok

    kl = torch.tensor(0.0, device=device)
    if cfg.kl_coef > 0:
        # k3 estimator of KL(pi_theta || pi_anchor) on sampled tokens
        # (DeepSeekMath's unbiased form). Anchor per cfg.kl_ref.
        if cfg.kl_ref == "init":
            logP_anchor = logP_ref if cfg.mode == "gopd" else _ref_logP(
                cfg, student, input_ids, attn, tgt, device)
        else:                                      # "teacher"
            logP_anchor = logP_T
        ratio = (logP_anchor - logP_S)
        kl = (((ratio.exp() - 1) - ratio) * tmask).sum() / ntok
    loss = pg_loss + cfg.kl_coef * kl

    if zero_grad:
        opt.zero_grad()
    (loss * loss_scale).backward()
    if opt_step:
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], cfg.grad_clip)
        opt.step()

    # ---- misaligned-mass probe: Corollary 1's margin over OPD is exactly
    # 2 * sum_{d_t*iota<0} |d_t*iota| -- this fraction decides if token-level pays ----
    probe = float("nan")
    if cfg.mode == "targeted":
        with torch.no_grad():
            di = (d_t * iota)[tmask]
            denom = di.abs().sum().clamp(min=1e-8)
            probe = (di[di < 0].abs().sum() / denom).item()

    return {"loss": loss.item(), "pg": pg_loss.item(),
            "reward_correct": A_vec.gt(0).float().mean().item(),
            "nondegen_frac": nondegen, "misaligned_mass": probe,
            "agree": agree_rate, "len_mean": len_mean, "cap_frac": cap_frac,
            "is_reject": is_reject, "is_w_mean": is_w_mean, "sample": _sample}


# ----------------------------------------------------------------------------- #
# Influence via the last-layer rank-one identity (paper eq 15), sign only.
#   G = sum_i A_i sum_t (e_{y_t} - pi_t) h_t^T          [|V| x h]  (PER-QUESTION
#       unnormalized sum over that question's own group -- eq 6 g_Q; the paper's
#       1/N rescales G and every LOO term equally, leaving sign(iota) unchanged)
#   iota(y_t) = ((G - G_i) h_t)_{y_t} - E_{pi_t}[(G - G_i) h_t]   (exact LOO)
# The per-rollout correction collapses to Gram matrices:
#   corr(t) = A_i * sum_s (h_s.h_t) * [1{y_s=y_t} - pi_s(y_t) - pi_t(y_s)
#                                       + <pi_s, pi_t>]
# whose s=t diagonal A_i |h_t|^2 |e_{y_t}-pi_t|^2 is Appendix B's bias term.
# Everything here is detached; only the sign feeds the (detached) reward.
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def compute_influence(cfg, s_logits, hidden, tgt, tmask, A_vec):
    # PER-QUESTION dispatch (fix 2026-07-13): the paper's G is the unembedding
    # block of ONE question's verifier PG (eq 6 g_Q), so every group of
    # group_size rollouts gets its own G, its own LOO and its own split-half.
    # The previous behavior built a single batch-level G across prompts
    # whenever prompts_per_step > 1 -- a deviation from Algorithm 1 and its
    # proofs. Rows are group-contiguous: question g occupies rows
    # [g*G, (g+1)*G) (see the rollout-collection loop in train_step).
    Gsz = cfg.group_size
    if s_logits.shape[0] <= Gsz:
        return _influence_one_group(cfg, s_logits, hidden, tgt, tmask, A_vec)
    iotas, alts = [], []
    for g0 in range(0, s_logits.shape[0], Gsz):
        io, al = _influence_one_group(
            cfg, s_logits[g0:g0 + Gsz], hidden[g0:g0 + Gsz],
            tgt[g0:g0 + Gsz], tmask[g0:g0 + Gsz], A_vec[g0:g0 + Gsz])
        iotas.append(io)
        alts.append(al)
    return torch.cat(iotas, 0), torch.cat(alts, 0)


@torch.no_grad()
def _influence_one_group(cfg, s_logits, hidden, tgt, tmask, A_vec):
    # ONE question's group. s_logits: [G, L-1, V], detached, any float dtype.
    # fp32 probabilities are materialized only per chunk (never the full-vocab
    # tensor). Returns (iota, iota_alt):
    #   iota     -- eq 15 from the group G, with exact per-rollout LOO
    #   iota_alt -- independent sign estimate from the OPPOSITE half of the
    #               group (split-half confidence test). Needs no LOO: the
    #               opposite half never contains the token's own rollout.
    #               (Verified against brute-force G construction, 0 error.)
    B, Lm1, V = s_logits.shape
    h = hidden.shape[-1]
    device = hidden.device

    idx = tmask.reshape(-1).nonzero(as_tuple=True)[0] # [N] completion-token slots
    N = idx.numel()
    if N == 0:
        z = torch.zeros(B, Lm1, device=device)
        return z, z.clone()
    bi = idx // Lm1                                   # rollout (= row) per token
    ti = idx % Lm1
    hid_f = hidden.reshape(-1, h)[idx].float()        # [N, h] (h small: cheap)
    tgt_f = tgt.reshape(-1)[idx]                      # [N]
    A_f = A_vec[bi]                                   # [N]
    half_f = (bi % cfg.group_size) % 2                # rollout's half within group

    def probs_at(sl):                                 # fp32 softmax, chunk only
        return F.softmax(s_logits[bi[sl], ti[sl]].float(), dim=-1)

    # ---- pass 1: accumulate G (and the two half-batch G's) ----
    G = torch.zeros(V, h, device=device, dtype=torch.float32)
    G_half = [torch.zeros(V, h, device=device, dtype=torch.float32),
              torch.zeros(V, h, device=device, dtype=torch.float32)]
    for s in range(0, N, cfg.influence_chunk):
        e = slice(s, min(s + cfg.influence_chunk, N))
        probs_c = probs_at(e)                         # [c, V]
        hc, Ac = hid_f[e], A_f[e]
        G -= (Ac.unsqueeze(1) * probs_c).t() @ hc     # -A * pi h^T
        G.index_add_(0, tgt_f[e], Ac.unsqueeze(1) * hc)  # +A * e_y h^T
        for hv in (0, 1):
            m = half_f[e] == hv
            if m.any():
                G_half[hv] -= (Ac[m].unsqueeze(1) * probs_c[m]).t() @ hc[m]
                G_half[hv].index_add_(0, tgt_f[e][m], Ac[m].unsqueeze(1) * hc[m])

    # ---- pass 2: iota (full G) and iota_alt (opposite half's G) ----
    iota_f = torch.empty(N, device=device, dtype=torch.float32)
    alt_f = torch.empty(N, device=device, dtype=torch.float32)
    for s in range(0, N, cfg.influence_chunk):
        e = slice(s, min(s + cfg.influence_chunk, N))
        probs_c = probs_at(e)
        Gh = hid_f[e] @ G.t()                          # [c, V]
        expect = (probs_c * Gh).sum(-1)                # E_pi[G h_t]
        yval = Gh.gather(1, tgt_f[e].unsqueeze(1)).squeeze(1)
        iota_f[e] = yval - expect
        Gh0 = hid_f[e] @ G_half[0].t()
        Gh1 = hid_f[e] @ G_half[1].t()
        opp = torch.where((half_f[e] == 0).unsqueeze(1), Gh1, Gh0)
        alt_f[e] = (opp.gather(1, tgt_f[e].unsqueeze(1)).squeeze(1)
                    - (probs_c * opp).sum(-1))

    # ---- pass 3: exact leave-one-out on iota (Algorithm 1 line 2) ----
    if cfg.leave_one_out:
        for i in range(B):
            sel = (bi == i).nonzero(as_tuple=True)[0]
            Ti = sel.numel()
            if Ti == 0 or A_vec[i].abs() < 1e-12:
                continue
            Hi = hid_f[sel]                            # [T, h]
            Pi = probs_at(sel)                         # [T, V]
            yi = tgt_f[sel]                            # [T]
            C = Hi @ Hi.t()                            # [T, T]  h_s . h_t
            M1 = Pi[:, yi]                             # M1[s, t] = pi_s(y_t)
            PP = Pi @ Pi.t()                           # <pi_s, pi_t>
            EQ = (yi.unsqueeze(1) == yi.unsqueeze(0)).float()
            K = EQ - M1 - M1.t() + PP                  # symmetric
            iota_f[sel] -= A_vec[i] * (C * K).sum(0)   # subtract G_i h_t part

    iota = torch.zeros(B * Lm1, device=device)
    iota[idx] = iota_f
    alt = torch.zeros(B * Lm1, device=device)
    alt[idx] = alt_f
    return iota.reshape(B, Lm1), alt.reshape(B, Lm1)


# ----------------------------------------------------------------------------- #
# Algorithm 2 (divergence view, reference-free): per visited state build the
# tilted target over the WHOLE vocabulary and descend the closed-form KL.
#   iota^(-i)(y) = ((G - G_i) h_t)_y - E_pi[(G - G_i) h_t]        (eq 15 + LOO)
#   G_i h_t collapses per rollout to A_i (S - M)[:,t], with
#     C = H_i H_i^T,  M = P_i^T C,  S[y_s,:] += C[s,:]            (verified vs
#   brute force to 1e-15; sampled-token entries match Algorithm 1's LOO).
#   mu_t = softmax(log pi_theta + sign(iota(.)) |d(.)|),  |d| <= tilt_clip
#   loss = (1/N) sum_t KL(pi_theta(.|s_t) || sg(mu_t))            (Algo 2 L6)
# Gradient flows through pi_theta's full softmax -- no sampling variance.
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def build_G(cfg, logits_flat, idx, hid_f, tgt_f, A_f, V, h, device):
    # G = sum_i A_i sum_t (e_{y_t} - pi_t) h_t^T, chunked over states
    G = torch.zeros(V, h, device=device, dtype=torch.float32)
    N = idx.numel()
    for s in range(0, N, cfg.influence_chunk):
        e = slice(s, min(s + cfg.influence_chunk, N))
        probs_c = F.log_softmax(logits_flat[idx[e]].float(), dim=-1).exp()
        hc, Ac = hid_f[e], A_f[e]
        G -= (Ac.unsqueeze(1) * probs_c).t() @ hc
        G.index_add_(0, tgt_f[e], Ac.unsqueeze(1) * hc)
    return G


def targeted_kl_loss(cfg, s_logits, t_logits, hidden, tgt, tmask, A_vec):
    B, Lm1, V = s_logits.shape
    # Same per-question-G requirement as compute_influence (eq 6 g_Q): this
    # loss builds ONE G over the forward batch, so the batch must be a single
    # question's group. Guard instead of silently mixing prompts.
    assert B <= cfg.group_size, (
        "targeted_kl builds a per-question G; run with prompts_per_step=1")
    h = hidden.shape[-1]
    device = s_logits.device

    sl_flat = s_logits.reshape(-1, V)              # view, WITH grad
    tl_flat = t_logits.reshape(-1, V)              # view, no grad
    idx = tmask.reshape(-1).nonzero(as_tuple=True)[0]
    N = idx.numel()
    zeros = torch.zeros(B, Lm1, device=device)
    if N == 0:
        return s_logits.sum() * 0.0, zeros, zeros.clone()
    hid_f = hidden.reshape(-1, h)[idx].float()     # [N, h]
    tgt_f = tgt.reshape(-1)[idx]
    rid_f = idx // Lm1                             # rollout (= row) per token

    G = build_G(cfg, sl_flat.detach(), idx, hid_f, tgt_f, A_vec[rid_f],
                V, h, device)

    loss_sum = None
    iota_samp = torch.zeros(B * Lm1, device=device)
    d_samp = torch.zeros(B * Lm1, device=device)
    for i in range(B):                             # rollout = natural state chunk
        sel = (rid_f == i).nonzero(as_tuple=True)[0]
        Ti = sel.numel()
        if Ti == 0:
            continue
        pos = idx[sel]
        yi = tgt_f[sel]
        logp_s = F.log_softmax(sl_flat[pos].float(), dim=-1)       # [T,V] grad
        with torch.no_grad():
            p_s = logp_s.detach().exp()
            logp_t = F.log_softmax(tl_flat[pos].float(), dim=-1)
            Hi = hid_f[sel]
            Gh = Hi @ G.t()                                        # [T,V]
            if cfg.leave_one_out and A_vec[i].abs() > 1e-12:
                C = Hi @ Hi.t()                                    # [T,T]
                M = p_s.t() @ C                                    # [V,T]
                S = torch.zeros(V, Ti, device=device)
                S.index_add_(0, yi, C)                             # S[y_s,:] += C[s,:]
                Gh = Gh - A_vec[i] * (S - M).t()                   # drop own rollout
            iota_vec = Gh - (p_s * Gh).sum(-1, keepdim=True)       # centered (eq 15)
            d_vec = (logp_t - logp_s.detach()).clamp(-cfg.tilt_clip, cfg.tilt_clip)
            mu_logits = logp_s.detach() + iota_vec.sign() * d_vec.abs()
            log_mu = F.log_softmax(mu_logits, dim=-1)              # sg(mu_t)
            iota_samp[pos] = iota_vec.gather(1, yi.unsqueeze(1)).squeeze(1)
            d_samp[pos] = d_vec.gather(1, yi.unsqueeze(1)).squeeze(1)
        kl_i = (logp_s.exp() * (logp_s - log_mu)).sum(-1)          # [T], grad
        loss_sum = kl_i.sum() if loss_sum is None else loss_sum + kl_i.sum()

    loss = loss_sum / N
    return loss, iota_samp.reshape(B, Lm1), d_samp.reshape(B, Lm1)


# ----------------------------------------------------------------------------- #
# Eval: greedy pass@1 on N test questions
# ----------------------------------------------------------------------------- #
_VLLM = {"llm": None, "dir": None, "counter": 0, "last_path": None}

def _ensure_vllm(cfg):
    # One co-located engine for BOTH eval and training rollouts. Base weights
    # load once; the LoRA adapter is hot-swapped via LoRARequest.
    import tempfile
    from vllm import LLM
    if _VLLM["llm"] is None:
        _VLLM["llm"] = LLM(
            model=cfg.student, enable_lora=True, max_lora_rank=32,
            gpu_memory_utilization=cfg.vllm_gpu_frac,
            max_model_len=cfg.max_prompt_tokens + max(cfg.eval_max_new_tokens,
                                                      cfg.max_new_tokens),
            enforce_eager=True)   # skip CUDA-graph capture: faster init,
                                  # less memory -- right call when co-located
                                  # with the HF training process
        _VLLM["dir"] = tempfile.mkdtemp(prefix="lora_hotswap_")
    return _VLLM["llm"]

def _fresh_adapter_request(cfg, student, reuse=False):
    # Save the CURRENT adapter (~35MB, 1-2s) under a new id; delete the
    # previous save so disk use stays bounded over a long run.
    # reuse=True skips the save and returns the last request -- valid within
    # one optimizer step (weights unchanged between accumulation micro-batches).
    import os
    import shutil
    from vllm.lora.request import LoRARequest
    if reuse and _VLLM.get("last_req") is not None:
        return _VLLM["last_req"]
    _VLLM["counter"] += 1
    path = os.path.join(_VLLM["dir"], f"step{_VLLM['counter']}")
    student.save_pretrained(path)
    if _VLLM["last_path"] and os.path.isdir(_VLLM["last_path"]):
        shutil.rmtree(_VLLM["last_path"], ignore_errors=True)
    _VLLM["last_path"] = path
    req = LoRARequest(f"ad{_VLLM['counter']}", _VLLM["counter"], path)
    _VLLM["last_req"] = req
    return req

def _vllm_rollouts(cfg, tok, student, prompt_texts, reuse_adapter=False):
    # Training rollouts through vLLM: returns, per prompt,
    # (prompt_token_ids, [(completion_token_ids, text), ...]) with n=G samples.
    # The policy is exact: the adapter saved THIS step is the current policy.
    from vllm import SamplingParams
    llm = _ensure_vllm(cfg)
    req = _fresh_adapter_request(cfg, student, reuse=reuse_adapter)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")   # dynamic: works for any
                                                       # Qwen family, not hard 151645
    sp = SamplingParams(temperature=cfg.temperature, top_p=cfg.top_p,
                        presence_penalty=cfg.presence_penalty,  # kill repeat loops
                        max_tokens=cfg.max_new_tokens, n=cfg.group_size,
                        stop_token_ids=[im_end],
                        # sampled-token logprob per step, for the TIS correction
                        logprobs=0 if cfg.rollout_is_correction else None,
                        # reproducible AND varies per optimizer step (counter bumps
                        # once per adapter save); micro-batches of a step share it
                        # but see different prompts.
                        seed=cfg.seed + _VLLM["counter"],
                        truncate_prompt_tokens=cfg.max_prompt_tokens)
    outs = llm.generate(prompt_texts, sp, lora_request=req, use_tqdm=False)

    def _samp_logps(o):                                # sampled-token logprob/token
        if not cfg.rollout_is_correction or o.logprobs is None:
            return None
        out = []
        for t, tid in enumerate(o.token_ids):
            d = o.logprobs[t] if t < len(o.logprobs) else None
            lp = d.get(tid) if d else None
            out.append(lp.logprob if lp is not None else 0.0)
        return out

    groups = []
    for out in outs:
        pids = list(out.prompt_token_ids)
        groups.append((pids, [(list(o.token_ids), o.text, _samp_logps(o))
                              for o in out.outputs]))
    return groups

def _vllm_eval(cfg, tok, student, test, log=None):
    from vllm import SamplingParams
    log = log or tqdm.write
    llm = _ensure_vllm(cfg)
    req = _fresh_adapter_request(cfg, student)
    prompts = [build_prompt(cfg, tok, ex["question"]) for ex in test]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    sp = SamplingParams(temperature=0.0, max_tokens=cfg.eval_max_new_tokens,
                        presence_penalty=cfg.presence_penalty,  # greedy still loops
                        stop_token_ids=[im_end])   # dynamic im_end, not hard 151645
    outs = llm.generate(prompts, sp, lora_request=req)
    correct = sum(verify(extract_pred(o.outputs[0].text), ex["gold"])
                  for o, ex in zip(outs, test))
    # ---- sample dump: first-N fixed questions, watch stop/length/answer drift ----
    ns = max(cfg.eval_show_samples, 0)
    for ex, o in zip(test[:ns], outs[:ns]):
        txt = o.outputs[0].text
        ids = o.outputs[0].token_ids
        pred = extract_pred(txt)
        ok = "OK" if verify(pred, ex["gold"]) else "XX"
        stopped = 151645 in ids                       # emitted <|im_end|>?
        body = txt if len(txt) <= 900 else txt[:600] + "\n  ...[中略]...\n" + txt[-300:]
        log(f"  --- sample [{ok}] gold={ex['gold']} pred={pred!r} "
            f"| {len(ids)} tok | stop={stopped} ---")
        log(f"  Q: {ex['question'][:160]}")
        log("  " + body.replace("\n", "\n  "))
    return correct / len(test)


@torch.no_grad()
def evaluate(cfg, tok, student, test, log=None):
    student.eval()
    if cfg.use_vllm_eval:
        try:
            acc = _vllm_eval(cfg, tok, student, test, log=log)
            student.train()
            return acc
        except ImportError:
            tqdm.write("[eval] vllm not installed -- falling back to batched "
                       "HF eval. `pip install vllm` for 10x faster evals.")
            cfg.use_vllm_eval = False   # don't retry every eval
    correct, done, nobox, ncap = 0, 0, 0, 0
    pbar = tqdm(total=len(test), desc="eval pass@1", leave=False, unit="q")
    for bstart in range(0, len(test), cfg.eval_batch):
        batch = test[bstart:bstart + cfg.eval_batch]
        prompts = [build_prompt(cfg, tok, ex["question"]) for ex in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=cfg.max_prompt_tokens).to(cfg.device)
        out = student.generate(**enc, do_sample=False,
                               max_new_tokens=cfg.eval_max_new_tokens,
                               pad_token_id=tok.pad_token_id)
        gen = out[:, enc.input_ids.shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        finished = (gen == tok.eos_token_id).any(dim=1).tolist()
        for ex, text, fin in zip(batch, texts, finished):
            correct += verify(extract_pred(text), ex["gold"])
            if cfg.dataset != "gsm8k":
                if extract_boxed(text) is None:
                    nobox += 1             # missing \boxed
                if not fin:
                    ncap += 1              # hit the token cap (true truncation)
        done += len(batch)
        pbar.update(len(batch))
        pbar.set_postfix(acc=f"{correct / done:.3f}",
                         nobox=f"{nobox}/{done}", cap=f"{ncap}/{done}")
    pbar.close()
    if cfg.dataset != "gsm8k" and nobox > 0:
        tqdm.write(f"[eval] no-boxed {nobox}/{len(test)}, cap-hit "
                   f"{ncap}/{len(test)} (cap={cfg.eval_max_new_tokens}): "
                   f"cap-hit≈truncation, no-boxed w/ EOS≈format failure")
    student.train()
    return correct / len(test)


# ----------------------------------------------------------------------------- #
# Checkpointing: LoRA adapter + optimizer + step/ptr/eval history.
# Resume restores the data pointer too -- the train order is seed-determined,
# so the question sequence continues exactly where it left off. (Sampling RNG
# is not restored: resumed rollouts are statistically, not bitwise, identical.)
# ----------------------------------------------------------------------------- #
def _ckpt_dir(cfg):
    # Variant isolation (2026-07-14 lesson): targeted runs encode their line-4
    # coefficients into the dir name, so coefficient sweeps can NEVER share a
    # dir (a shared dir once let resume load a different variant's weights).
    # Default -> targeted_asym1.25-0.25_{dataset}_seed{seed}.
    # Other modes (opd/gopd/...) are coefficient-free and keep plain names.
    import os
    tag = cfg.mode
    if cfg.mode == "targeted":
        tag = f"targeted_asym{cfg.amp_same:g}-{cfg.amp_flip:g}"
    # run_prefix encodes the model/setup so different bases never share a dir.
    return os.path.join(cfg.save_dir, f"{cfg.run_prefix}_{tag}_{cfg.dataset}_seed{cfg.seed}")

def save_ckpt(cfg, student, opt, step, ptr, history):
    # Every save goes to its own step{N}/ subdir -- ALL checkpoints are kept
    # (adapter ~35MB + optimizer state ~70MB per snapshot), enabling later
    # per-checkpoint analysis / validation-based selection.
    import os
    try:
        d = os.path.join(_ckpt_dir(cfg), f"step{step}")
        os.makedirs(d, exist_ok=True)
        student.save_pretrained(os.path.join(d, "adapter"))
        torch.save({"opt": opt.state_dict(), "step": step, "ptr": ptr,
                    "history": history}, os.path.join(d, "state.pt"))
        tqdm.write(f"[ckpt] saved step {step} -> {d}")
    except OSError as e:
        tqdm.write(f"[ckpt] SAVE FAILED ({e}) -- is Drive mounted? "
                   f"Training continues without checkpoints.")

def try_resume(cfg, student, opt):
    # Resume from the LATEST valid step{N}/ snapshot under this run's dir.
    import os
    base = _ckpt_dir(cfg)
    if not (cfg.resume and os.path.isdir(base)):
        return 0, 0, []
    steps_found = []
    for name in os.listdir(base):
        m = re.match(r"step(\d+)$", name)
        if m:
            d = os.path.join(base, name)
            if (os.path.isfile(os.path.join(d, "state.pt")) and
                    os.path.isfile(os.path.join(d, "adapter",
                                                "adapter_model.safetensors"))):
                steps_found.append(int(m.group(1)))
    if not steps_found:
        return 0, 0, []
    d = os.path.join(base, f"step{max(steps_found)}")
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(student, load_file(
        os.path.join(d, "adapter", "adapter_model.safetensors")))
    st = torch.load(os.path.join(d, "state.pt"), map_location="cpu")
    opt.load_state_dict(st["opt"])
    print(f"[resume] restored {cfg.mode} at step {st['step']} from {d}")
    return st["step"], st["ptr"], st.get("history", [])


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def run(cfg: Config):
    random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    # ---- txt log: mirror every step/eval line to a file under the ckpt dir ----
    import os
    _logf = None
    if cfg.save_dir:
        os.makedirs(_ckpt_dir(cfg), exist_ok=True)
        _logf = open(os.path.join(_ckpt_dir(cfg), "train_log.txt"), "a")
    def LOG(msg):
        tqdm.write(msg)
        if _logf:
            _logf.write(msg + "\n"); _logf.flush()
    LOG(f"=== run {cfg.mode} | student={cfg.student} | "
        f"pps={cfg.prompts_per_step} ga={cfg.grad_accum} G={cfg.group_size} "
        f"lr={cfg.lr} steps={cfg.steps} eval_n={cfg.eval_n} ===")

    tok, student, teacher = load_models(cfg)

    train, test = load_data(cfg)

    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay)

    start_step, ptr, history = 0, 0, []
    if cfg.save_dir:
        start_step, ptr, history = try_resume(cfg, student, opt)

    print(f"\n=== mode={cfg.mode} ===")
    last_eval = None
    # eval_every <= 0 => train-only, no in-loop eval at all (evaluate later from
    # saved checkpoints). Skips step-0, periodic, and final evals.
    if start_step == 0 and cfg.eval_every > 0:
        base_acc = evaluate(cfg, tok, student, test, log=LOG)
        history.append((0, base_acc))
        LOG(f"[step 0] pass@1={base_acc:.3f}")

    pbar = tqdm(range(start_step + 1, cfg.steps + 1), desc=f"train[{cfg.mode}]",
                initial=start_step, total=cfg.steps)
    for step in pbar:
        lr_t = _lr_at(cfg, step)                  # warmup then CONSTANT (no decay)
        for g_ in opt.param_groups:
            g_["lr"] = lr_t
        # gradient accumulation: grad_accum micro-batches -> one optimizer
        # step. One "step" here consumes prompts_per_step*grad_accum questions.
        micro_logs = []
        for mi in range(cfg.grad_accum):
            qs, gs = [], []
            for _ in range(cfg.prompts_per_step):
                ex = train[ptr % len(train)]; ptr += 1
                qs.append(ex["question"]); gs.append(ex["gold"])
            micro_logs.append(train_step(
                cfg, tok, student, teacher, opt, qs, gs,
                zero_grad=(mi == 0),
                opt_step=(mi == cfg.grad_accum - 1),
                loss_scale=1.0 / cfg.grad_accum))
        # average metrics over micro-batches (nan-safe: agree/mis can be nan
        # on degenerate micros)
        log = {}
        for k in micro_logs[-1]:
            if k == "sample":                        # dict/None, not numeric: skip avg
                log[k] = micro_logs[0].get("sample")
                continue
            vals = [m[k] for m in micro_logs
                    if k in m and m[k] == m[k]]      # drop NaNs
            log[k] = sum(vals) / len(vals) if vals else float("nan")
        pbar.set_postfix(loss=f"{log['loss']:.3f}",
                         acc=f"{log['reward_correct']:.2f}",
                         nondegen=f"{log['nondegen_frac']:.2f}",
                         mis=f"{log['misaligned_mass']:.3f}",
                         agree=f"{log.get('agree', float('nan')):.2f}",
                         len=f"{log['len_mean']:.0f}",
                         cap=f"{log['cap_frac']:.2f}")
        if step % 10 == 0:
            LOG(f"[step {step}] loss={log['loss']:.4f} "
                f"acc_in_batch={log['reward_correct']:.2f} "
                f"nondegen={log['nondegen_frac']:.2f} "
                f"misaligned_mass={log['misaligned_mass']:.3f} "
                f"agree={log.get('agree', float('nan')):.2f} "
                f"len={log['len_mean']:.0f} cap={log['cap_frac']:.2f} "
                f"lr={lr_t:.1e}")
        if cfg.rollout_print_every > 0 and step % cfg.rollout_print_every == 0:
            s = micro_logs[0].get("sample")
            if s:
                body = (s["text"] if len(s["text"]) <= 800
                        else s["text"][:500] + "\n...[中略]...\n" + s["text"][-250:])
                LOG(f"[step {step}] ROLLOUT stop={s['stop']} len={s['len']}\n{body}")
        if cfg.eval_every > 0 and step % cfg.eval_every == 0:
            last_eval = evaluate(cfg, tok, student, test, log=LOG)
            history.append((step, last_eval))
            LOG(f"[step {step}] >>> pass@1={last_eval:.3f}")
        if cfg.save_dir and step % cfg.save_every == 0:
            save_ckpt(cfg, student, opt, step, ptr, history)

    # greedy eval is deterministic: if the last step just evaluated, reuse it
    if cfg.eval_every <= 0:
        final = None                              # train-only: no final eval
    elif cfg.steps % cfg.eval_every == 0 and last_eval is not None:
        final = last_eval
    else:
        final = evaluate(cfg, tok, student, test, log=LOG)
        history.append((cfg.steps, final))
    if cfg.save_dir:
        save_ckpt(cfg, student, opt, cfg.steps, ptr, history)
    LOG(f"[final] mode={cfg.mode} pass@1="
        + (f"{final:.3f}" if final is not None else "eval off (train-only)"))
    LOG("eval history: " + str(history))
    if _logf:
        _logf.close()
    return final
