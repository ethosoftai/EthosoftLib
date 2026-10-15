import sys, json, re, glob, os, random, warnings, collections
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from ethosoftlib.math.converter import convert_text, _CODE_HINT
random.seed(11)
show = sys.argv[1:]
NUM_EXPR = re.compile(r"\d\s*(?:\\times|\\cdot|\\div|[*/×÷+\-^])\s*\d|\\frac\{\d+\}\{\d+\}|\d\^\{?\d")
tot = collections.Counter()
for path in sorted(glob.glob("samples/*.jsonl")):
    name = os.path.basename(path)[:-6]
    rows = [json.loads(l)["text"] for l in open(path, encoding="utf-8")]
    c = collections.Counter(); ex = collections.defaultdict(list)
    for t in rows:
        if _CODE_HINT.search(t):
            c["code"] += 1; continue
        o, st = convert_text(t)
        if st["converted"] > 0:
            c["covered"] += 1; continue
        if NUM_EXPR.search(t):
            k = "num_expr_missed"
        elif "=" in t:
            k = "eq_nonnumeric"
        else:
            k = "no_computation"
        c[k] += 1; ex[k].append(t)
    n = len(rows)
    print(f"{name:20s} n={n} " + " ".join(f"{k}={v}" for k, v in sorted(c.items())))
    if name in show:
        for k in ("num_expr_missed", "eq_nonnumeric"):
            for t in random.sample(ex[k], min(5, len(ex[k]))):
                print(f"   [{k}]", t[:300].replace("\n", " ⏎ "))
