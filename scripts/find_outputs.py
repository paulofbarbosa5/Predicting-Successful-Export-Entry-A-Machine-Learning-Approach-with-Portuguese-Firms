import sys, os, csv

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join("scripts", "outputs")
META = ("method","model","model_name","sample","outcome","split_design","split",
        "feature_set","best_params","params","chosen","status","roc_auc","pr_auc",
        "lift_at_5","lift_at_10","precision_at_10","test_events","n_test")

print("=== ARVORE DE OUTPUTS ===")
for d, _, files in os.walk(ROOT):
    for f in sorted(files):
        p = os.path.join(d, f)
        try: print(f"{os.path.getsize(p)//1024:>8} KB  {p}")
        except OSError: pass

print("\n=== FICHEIROS COM best_params OU roc_auc ===")
for d, dirs, files in os.walk(ROOT):
    if os.path.basename(d).lower() == "predictions":
        dirs[:] = []; continue                      # nao abre previsoes por firma
    for f in sorted(files):
        if not f.lower().endswith((".csv", ".json")): continue
        p = os.path.join(d, f)
        try:
            if os.path.getsize(p) > 20_000_000: continue
            txt = open(p, encoding="utf-8", errors="ignore").read()
        except OSError: continue
        if "best_params" not in txt and "roc_auc" not in txt: continue
        print(f"\n--- {p}")
        if f.lower().endswith(".json"):
            print(txt[:4000]); continue
        rows = list(csv.DictReader(txt.splitlines()))
        if not rows: continue
        keep = [c for c in rows[0] if c in META] or list(rows[0])[:8]
        for r in rows[:80]:
            print(" | ".join(f"{c}={r.get(c,'')}" for c in keep))