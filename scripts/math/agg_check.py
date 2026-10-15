import sys, json, random, re, glob, os, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from ethosoftlib.math.converter import convert_text, _TAG_RE

folder = sys.argv[1] if len(sys.argv) > 1 else "samples_ho"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
random.seed(4)
items = []
for path in sorted(glob.glob(f"{folder}/*.jsonl")):
    name = os.path.basename(path)[:-6]
    rows = [json.loads(l)["text"] for l in open(path, encoding="utf-8")]
    random.shuffle(rows)
    for t in rows[:60]:
        a, _ = convert_text(t)
        b, _ = convert_text(t, aggressive=True)
        ca = {m.group(0) for m in _TAG_RE.finditer(a)}
        for m in _TAG_RE.finditer(b):
            if m.group(0) not in ca:
                s = max(0, m.start() - 45)
                items.append((name, b[s:m.start()].replace("\n", " ⏎ ") + " <<" + m.group(1) + " -> " + m.group(2) + ">> " + b[m.end():m.end() + 30].replace("\n", " ⏎ ")))
random.shuffle(items)
print(len(items), "aggressive-only calls in sample")
for name, ctx in items[:N]:
    print(f"[{name}] {ctx}")
