import sys, json, random, re, warnings
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
from ethosoftlib.math.converter import convert_text, verify_converted, _CODE_HINT, _ASY

folder, name = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
MAXLEN = int(sys.argv[4]) if len(sys.argv) > 4 else 320
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1
rows = [json.loads(l)["text"] for l in open(f"{folder}/{name}.jsonl", encoding="utf-8")]
unc = []
for t in rows:
    if len(t) > MAXLEN or _CODE_HINT.search(_ASY.sub("", t)) or not re.search(r"\d", t):
        continue
    if verify_converted(convert_text(t)[0])[0] == 0:
        unc.append(t)
random.seed(seed)
print(f"{name}: {len(unc)} short uncovered rows")
for t in random.sample(unc, min(N, len(unc))):
    print("---", t.replace("\n", " ⏎ "))
