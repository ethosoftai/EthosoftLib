"""Bir ornek klasorunu olcer ve kapsanmayan satirlari ozelliklerine gore siniflar.

kullanim: python analyze.py samples_ho [dataset_adi ...] [--show N]
"""
import sys, json, re, glob, os, random, warnings, collections, time
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from ethosoftlib.math.converter import convert_text, verify_converted, _CODE_HINT, _ASY

AGG = "--aggressive" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
SHOW = 0
if "--show" in sys.argv:
    SHOW = int(sys.argv[sys.argv.index("--show") + 1])
    args = [a for a in args if a != str(SHOW)]
folder = args[0] if args else "samples_ho"
only = args[1:]

FEATURES = [
    ("ineq", re.compile(r"\\le(?:q|slant)?\b|\\ge(?:q|slant)?\b|\\lt|\\gt|[^-=<]<[^<=]|[^->=]>[^>=]|\\neq|\\ne\b")),
    ("sum_prod", re.compile(r"\\sum|\\prod|\\cdots|\\ldots|\.\.\.")),
    ("deriv", re.compile(r"f'|\\frac\{d|\\partial|'\(x\)|\\prime")),
    ("integral", re.compile(r"\\int")),
    ("limit", re.compile(r"\\lim")),
    ("tuple_pt", re.compile(r"\(\s*-?[\d.]+\s*,\s*-?[\d.]+")),
    ("base_n", re.compile(r"\d_\{?\d+\}?|_\{?\d+\}?\)?\b")),
    ("modular", re.compile(r"\\pmod|\\equiv|\\bmod|\bmod\b|\\%")),
    ("subscript", re.compile(r"[A-Za-z]_\{?[A-Za-z0-9]+\}?")),
    ("matrix", re.compile(r"pmatrix|bmatrix|matrix")),
    ("complex", re.compile(r"\bi\^|\\sqrt\{-|\bi\b.*\bi\b|\\overline\{z")),
    ("vector", re.compile(r"\\mathbf|\\vec|\\overrightarrow")),
    ("trig_sym", re.compile(r"\\(?:sin|cos|tan|cot|sec|csc)\b")),
    ("log_exp", re.compile(r"\\(?:log|ln)\b|e\^")),
    ("approx_text", re.compile(r"approximately|about|\\approx|roughly|nearest")),
    ("words_num", re.compile(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twelve|twenty|hundred)\b", re.I)),
    ("percent", re.compile(r"\d\s*\\?%|percent")),
    ("proof_words", re.compile(r"\b(?:prove|proof|show that|suppose|contradiction|induction|lemma)\b", re.I)),
    ("mcq", re.compile(r"\\boxed\{\s*\(?[A-E]\)?\s*\}|The answer is \(?[A-E]\)?|answer is [A-E]\b|correct (?:option|choice|answer)", re.I)),
]
NUM_EXPR = re.compile(r"\d\s*(?:\\times|\\cdot|\\div|[*/×÷+\-^])\s*\d|\\frac\{\d+\}\{\d+\}|\d\^\{?\d")

total = collections.Counter()
feat_total = collections.Counter()
uncovered_dump = []
t0 = time.time()
for path in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
    name = os.path.basename(path)[:-6]
    if only and name not in only:
        continue
    rows = [json.loads(l)["text"] for l in open(path, encoding="utf-8")]
    c = collections.Counter()
    bad = calls = 0
    feats = collections.Counter()
    for t in rows:
        if _CODE_HINT.search(_ASY.sub("", t)):
            c["code"] += 1
            continue
        o, st = convert_text(t, aggressive=AGG)
        n, b = verify_converted(o)
        calls += n; bad += b
        if n > 0:
            c["covered"] += 1
            continue
        has_num = bool(re.search(r"\d", t)) and bool(NUM_EXPR.search(t) or "=" in t)
        if not has_num:
            c["no_computation"] += 1
            continue
        c["uncovered"] += 1
        uncovered_dump.append({"dataset": name, "text": t})
        for fname, rx in FEATURES:
            if rx.search(t):
                feats[fname] += 1
    n_rows = len(rows)
    addr = n_rows - c["code"] - c["no_computation"]
    pct_all = 100 * c["covered"] / max(1, n_rows)
    pct_addr = 100 * c["covered"] / max(1, addr)
    print(f"{name:20s} n={n_rows:5d} covered={c['covered']:5d} ({pct_all:3.0f}%)  code={c['code']:4d} "
          f"nocomp={c['no_computation']:4d} uncovered={c['uncovered']:4d}  addressable_cov={pct_addr:3.0f}%  "
          f"calls={calls} bad={bad}")
    total.update(c)
    feat_total.update(feats)
n = sum(total.values())
addr = n - total["code"] - total["no_computation"]
print(f"\nTOPLAM n={n} covered={total['covered']} ({100*total['covered']/max(1,n):.1f}%)  "
      f"code={total['code']} nocomp={total['no_computation']} uncovered={total['uncovered']}  "
      f"addressable_cov={100*total['covered']/max(1,addr):.1f}%  ({time.time()-t0:.0f}s)")
print("Kapsanmayan satirlarda ozellik sayilari:", dict(feat_total.most_common()))
out = f"unc_{os.path.basename(folder.rstrip('/'))}.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for r in uncovered_dump:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
if SHOW:
    random.seed(13)
    for r in random.sample(uncovered_dump, min(SHOW, len(uncovered_dump))):
        print(f"\n[{r['dataset']}]", r["text"][:420].replace("\n", " ⏎ "))
