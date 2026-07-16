# -*- coding: utf-8 -*-
"""判分器:与 exp-0715 训练脚本逐行一致(\\boxed{} 嵌套解析 → math-verify →
规范化字符串/浮点回退),保证探针测的是训练同款验证器。"""
import re

_num = re.compile(r"-?\d[\d,]*\.?\d*")

try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    HAVE_MATH_VERIFY = True
except Exception:
    HAVE_MATH_VERIFY = False


def extract_boxed(text: str):
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
    return None  # 括号不闭合(截断)


def extract_pred(text: str) -> str:
    b = extract_boxed(text)
    if b is not None:
        return b.strip()
    if "####" in text:
        tail = text.split("####")[-1]
        m = _num.search(tail)
        if m:
            return m.group(0).replace(",", "")
    nums = _num.findall(text)
    return nums[-1].replace(",", "") if nums else ""


def _norm(s: str) -> str:
    s = s.strip().strip("$").replace(" ", "").replace(",", "")
    for t in ("\\left", "\\right", "\\!", "\\,"):
        s = s.replace(t, "")
    return s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")


def verify(pred: str, gold: str) -> float:
    if not pred:
        return 0.0
    if HAVE_MATH_VERIFY:
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
