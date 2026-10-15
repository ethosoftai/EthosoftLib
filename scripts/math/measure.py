import sys, json, glob, os, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from ethosoftlib.math.converter import convert_text, verify_converted

tot_rows = tot_cov = 0
for path in sorted(glob.glob("samples/*.jsonl")):
    name = os.path.basename(path)[:-6]
    rows = [json.loads(l)["text"] for l in open(path, encoding="utf-8")]
    cov = calls = bad = mism = 0
    for t in rows:
        o, st = convert_text(t)
        n, b = verify_converted(o)
        cov += n > 0
        calls += n
        bad += b
        mism += st["mismatch"]
    tot_rows += len(rows); tot_cov += cov
    print(f"{name:20s} rows={len(rows):4d} with_tool={cov:4d} ({100*cov/len(rows):3.0f}%) calls={calls:5d} verify_bad={bad} skipped_wrong={mism}")
print(f"TOPLAM satir={tot_rows} aracli={tot_cov} ({100*tot_cov/tot_rows:.0f}%)")
