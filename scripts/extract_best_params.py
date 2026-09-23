import sys, os, csv
ROOT = sys.argv[1] if len(sys.argv) > 1 else ".."
SKIP = {".git","__pycache__",".venv","venv","env",".ipynb_checkpoints","raw","processed","models","predictions"}

def walk():
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in SKIP]
        for f in files: yield d, f, os.path.join(d, f)

print("=== 1. GRELHAS COMPLETAS: _model_common.py linhas 185-245 ===")
for d, f, p in walk():
    if f == "_model_common.py":
        for i, l in enumerate(open(p, encoding="utf-8", errors="ignore").read().splitlines()[184:245], 185):
            print(f"{i:4}: {l}")
        break

print("\n=== 2. best_params GRAVADOS NOS OUTPUTS ===")
for d, f, p in walk():
    if not f.lower().endswith((".csv", ".json", ".txt")): continue
    try:
        if os.path.getsize(p) > 5_000_000: continue
        txt = open(p, encoding="utf-8", errors="ignore").read()
    except Exception: continue
    if "best_params" not in txt: continue
    print(f"\n--- {p}")
    if f.lower().endswith(".csv"):
        rows = list(csv.DictReader(txt.splitlines()))
        keep = [c for c in (rows[0] if rows else {}) if c in
                ("method","model","model_name","sample","outcome","split_design",
                 "feature_set","best_params","roc_auc","pr_auc","lift_at_10","status")]
        for r in rows[:60]: print(" | ".join(f"{c}={r.get(c,'')}" for c in keep))
    else:
        for l in txt.splitlines():
            if "best_params" in l: print(l.strip()[:300])

print("\n=== 3. QUE SPLITS TEM METRICAS CALCULADAS ===")
for d, f, p in walk():
    if not f.lower().endswith(".csv"): continue
    try:
        rows = list(csv.DictReader(open(p, encoding="utf-8", errors="ignore")))
    except Exception: continue
    if not rows: continue
    cols = rows[0].keys()
    sc = next((c for c in cols if "split" in c.lower()), None)
    if sc and any("auc" in c.lower() for c in cols):
        print(f"{p}: {sorted({r[sc] for r in rows if r.get(sc)})}")