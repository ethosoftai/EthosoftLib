"""Ayar yaptigimiz ilk satirlardan FARKLI satirlari (skip=1000) indirir: samples_ho/*.jsonl"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from datasets import load_dataset

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SKIP = 1000
SPECS = [
    ("gsm8k", dict(path="openai/gsm8k", name="main", split="train"), "answer"),
    ("MATH-algebra", dict(path="EleutherAI/hendrycks_math", name="algebra", split="train"), "solution"),
    ("MATH-geometry", dict(path="EleutherAI/hendrycks_math", name="geometry", split="train"), "solution"),
    ("MATH-prob", dict(path="EleutherAI/hendrycks_math", name="counting_and_probability", split="train"), "solution"),
    ("MATH-numtheory", dict(path="EleutherAI/hendrycks_math", name="number_theory", split="train"), "solution"),
    ("MATH-intalg", dict(path="EleutherAI/hendrycks_math", name="intermediate_algebra", split="train"), "solution"),
    ("MATH-prealg", dict(path="EleutherAI/hendrycks_math", name="prealgebra", split="train"), "solution"),
    ("MATH-precalc", dict(path="EleutherAI/hendrycks_math", name="precalculus", split="train"), "solution"),
    ("MetaMathQA", dict(path="meta-math/MetaMathQA", split="train"), "response"),
    ("orca-math", dict(path="microsoft/orca-math-word-problems-200k", split="train"), "answer"),
    ("NuminaMath-CoT", dict(path="AI-MO/NuminaMath-CoT", split="train"), "solution"),
    ("NuminaMath-TIR", dict(path="AI-MO/NuminaMath-TIR", split="train"), "solution"),
    ("OpenMathInstruct-2", dict(path="nvidia/OpenMathInstruct-2", split="train_1M"), "generated_solution"),
    ("MathInstruct", dict(path="TIGER-Lab/MathInstruct", split="train"), "output"),
    ("OpenR1-solution", dict(path="open-r1/OpenR1-Math-220k", name="default", split="train"), "solution"),
    ("DeepScaleR", dict(path="agentica-org/DeepScaleR-Preview-Dataset", split="train"), "solution"),
    ("AQuA-RAT", dict(path="deepmind/aqua_rat", name="raw", split="train"), "rationale"),
    ("DeepMath-r1", dict(path="zwhe99/DeepMath-103K", split="train"), "r1_solution_1"),
]
os.makedirs("samples_ho", exist_ok=True)
only = sys.argv[2:]
for name, kw, field in SPECS:
    if only and name not in only:
        continue
    path = f"samples_ho/{name}.jsonl"
    if os.path.exists(path) and sum(1 for _ in open(path, encoding="utf-8")) >= min(N, 500):
        print("skip", name, flush=True)
        continue
    try:
        ds = load_dataset(streaming=True, **kw).skip(SKIP)
        rows = []
        for i, r in enumerate(ds):
            if i >= N:
                break
            if isinstance(r.get(field), str):
                rows.append({"text": r[field]})
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("ok", name, len(rows), flush=True)
    except Exception as e:
        print("ERR", name, str(e)[:120], flush=True)
