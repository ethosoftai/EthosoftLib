import sys, time, json, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from datasets import load_dataset
from ethosoftlib.math.converter import convert_text, verify_converted

N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 300
SPECS = [
    ("gsm8k", dict(path="openai/gsm8k", name="main", split="train"), "answer"),
    ("MATH-algebra", dict(path="EleutherAI/hendrycks_math", name="algebra", split="train"), "solution"),
    ("MATH-geometry", dict(path="EleutherAI/hendrycks_math", name="geometry", split="train"), "solution"),
    ("MATH-prob", dict(path="EleutherAI/hendrycks_math", name="counting_and_probability", split="train"), "solution"),
    ("MetaMathQA", dict(path="meta-math/MetaMathQA", split="train"), "response"),
    ("orca-math", dict(path="microsoft/orca-math-word-problems-200k", split="train"), "answer"),
    ("NuminaMath-CoT", dict(path="AI-MO/NuminaMath-CoT", split="train"), "solution"),
    ("OpenMathInstruct-2", dict(path="nvidia/OpenMathInstruct-2", split="train_1M"), "generated_solution"),
    ("MathInstruct", dict(path="TIGER-Lab/MathInstruct", split="train"), "output"),
]

def main():
    only = sys.argv[2:]
    rows_out = []
    for name, kw, field in SPECS:
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            ds = load_dataset(streaming=True, **kw)
            agg = dict(rows=0, rows_conv=0, found=0, conv=0, mism=0, vt=0, vbad=0)
            ex = []
            for i, r in enumerate(ds):
                if i >= N:
                    break
                txt = r[field]
                if not isinstance(txt, str):
                    continue
                out, st = convert_text(txt)
                vt, vb = verify_converted(out)
                agg["rows"] += 1
                agg["rows_conv"] += st["converted"] > 0
                agg["found"] += st["found"]
                agg["conv"] += st["converted"]
                agg["mism"] += st["mismatch"]
                agg["vt"] += vt
                agg["vbad"] += vb
                if st["converted"] and len(ex) < 2:
                    ex.append((txt, out))
            print(f"== {name}: rows={agg['rows']} rows_with_tool={agg['rows_conv']} "
                  f"({100*agg['rows_conv']/max(1,agg['rows']):.0f}%) eq_found={agg['found']} "
                  f"converted={agg['conv']} mismatch(skipped)={agg['mism']} "
                  f"verify={agg['vt']-agg['vbad']}/{agg['vt']} ({time.time()-t0:.0f}s)")
            for a, b in ex[:1]:
                print("   IN :", a[:380].replace("\n", " ⏎ "))
                print("   OUT:", b[:520].replace("\n", " ⏎ "))
        except Exception as e:
            print(f"== {name}: HATA {type(e).__name__}: {str(e)[:160]}")

if __name__ == '__main__':
    main()
