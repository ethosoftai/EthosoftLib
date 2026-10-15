"""Her dataset'ten N satiri yerel jsonl'ye indirir (olcumleri hizli tekrarlamak icin)."""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from datasets import load_dataset
from test_samples import SPECS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
os.makedirs("samples", exist_ok=True)
for name, kw, field in SPECS:
    path = f"samples/{name}.jsonl"
    if os.path.exists(path) and sum(1 for _ in open(path, encoding="utf-8")) >= N:
        print("skip", name, flush=True)
        continue
    try:
        ds = load_dataset(streaming=True, **kw)
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
