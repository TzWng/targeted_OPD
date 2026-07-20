# ==== eval: raw vs sft 两轮 opd checkpoints(各自正确 base)====
import os, re, gc, glob, shutil, torch
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD","spawn")
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

BASE="/content/drive/MyDrive/OPD"; TMP="/content/local_models/_merged"
SYS="Please reason step by step, and put your final answer within \\boxed{}."
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B"); IM_END=tok.convert_tokens_to_ids("<|im_end|>")

RUNS = {
    "raw": {"dir": f"{BASE}/q3-1.7b_opd_deepmath_seed0",     "base": "Qwen/Qwen3-1.7B-Base"},
}
ONLY_STEPS = [50, 100, 150, 200, 250, 300, 350, 400]   # 空列表=全测;先粗扫这几个

def eb(t):
    i=t.rfind("\\boxed")
    if i==-1: return None
    j=t.find("{",i)
    if j==-1: return None
    d=0
    for k in range(j,len(t)):
        if t[k]=="{":d+=1
        elif t[k]=="}":
            d-=1
            if d==0:return t[j+1:k]
    return None
def ep(t):
    b=eb(t)
    if b is not None: return b.strip()
    n=re.findall(r"-?\d[\d,]*\.?\d*",t); return n[-1].replace(",","") if n else ""
try:
    from math_verify import parse as _mvp, verify as _mvv; _HMV=True
except Exception: _HMV=False
def okv(p,g):
    if not p: return False
    if _HMV:
        try: return bool(_mvv(_mvp(str(g)),_mvp(str(p))))
        except Exception: pass
    p=str(p).replace(" ","").replace(",",""); g=str(g).replace(" ","").replace(",","")
    if p==g: return True
    try: return abs(float(p)-float(g))<1e-4
    except: return False
def build_p(q):
    m=[{"role":"system","content":SYS},{"role":"user","content":q}]
    try: return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
def free_gpu():
    try:
        import psutil
        kids=psutil.Process(os.getpid()).children(recursive=True)
        for ch in kids:
            try: ch.kill()
            except: pass
        psutil.wait_procs(kids,timeout=5)
    except Exception: pass
    gc.collect(); torch.cuda.empty_cache()

def tl(n,fn):
    try: xs=fn(); print(f"  {n}:{len(xs)}"); return xs
    except Exception as e: print(f"  {n} FAIL {repr(e)[:80]}"); return None
benches={}
benches["MATH500"]=tl("MATH500",lambda:[{"q":x["problem"],"g":x["answer"]}
    for x in load_dataset("json",data_files=f"{BASE}/MATH500/test.jsonl",split="train")])
benches["Olympiad"]=tl("Olympiad",lambda:[{"q":x["question"],"g":str(x["final_answer"][0])}
    for x in load_dataset("Hothan/OlympiadBench","OE_TO_maths_en_COMP",split="train") if x.get("final_answer")])
benches["Minerva"]=tl("Minerva",lambda:[{"q":x["question"],"g":x["answer"]}
    for x in load_dataset("math-ai/minervamath",split="test")])
benches={k:v for k,v in benches.items() if v}

def score(path):
    llm=LLM(model=path, tokenizer="Qwen/Qwen3-1.7B", gpu_memory_utilization=0.6,
            max_model_len=4096, enforce_eager=True)
    row={}
    for bn,items in benches.items():
        outs=llm.generate([build_p(it["q"]) for it in items],
            SamplingParams(temperature=0,presence_penalty=1.5,max_tokens=2048,stop_token_ids=[IM_END]))
        row[bn]=sum(okv(ep(o.outputs[0].text),it["g"]) for o,it in zip(outs,items))/len(items)
    del llm; return row

res={}
for name,info in RUNS.items():
    d,b=info["dir"],info["base"]
    if not os.path.isdir(d): print(f"跳过 {name}: 无目录 {d}"); continue
    steps=sorted(int(re.search(r"step(\d+)/adapter",p).group(1))
                 for p in glob.glob(f"{d}/step*/adapter/adapter_model.safetensors"))
    if ONLY_STEPS: steps=[s for s in steps if s in ONLY_STEPS]
    print(f"\n===== {name}: base={b} | steps={steps} =====")
    try:
        res[(name,0)]=score(b); print("  base:",{k:round(v,3) for k,v in res[(name,0)].items()})
    except Exception as e: print(f"  base FAIL {repr(e)[:120]}")
    free_gpu()
    for s in steps:
        try:
            if os.path.isdir(TMP): shutil.rmtree(TMP)
            mb=AutoModelForCausalLM.from_pretrained(b,dtype=torch.bfloat16)
            PeftModel.from_pretrained(mb,f"{d}/step{s}/adapter").merge_and_unload().save_pretrained(TMP,safe_serialization=True)
            tok.save_pretrained(TMP); del mb; gc.collect(); torch.cuda.empty_cache()
            res[(name,s)]=score(TMP); print(f"  step{s}:",{k:round(v,3) for k,v in res[(name,s)].items()})
        except Exception as e: print(f"  step{s} FAIL {repr(e)[:120]}")
        free_gpu()

# 每个 run 一张表(teacher=Qwen3-4B 参照:0.726/0.405/0.261)
for name,info in RUNS.items():
    ss=sorted({s for (n,s) in res if n==name})
    if not ss: continue
    print(f"\n### {name} (base={info['base']}) ###")
    print(f"{'step':>7s}"+"".join(f"{bn:>11s}" for bn in benches))
    for s in ss:
        print(f"{('base' if s==0 else s):>7}"+"".join(f"{res[(name,s)].get(bn,float('nan')):>11.3f}" for bn in benches))
    print(f"{'teacher':>7}{0.726:>11.3f}{0.405:>11.3f}{0.261:>11.3f}")
