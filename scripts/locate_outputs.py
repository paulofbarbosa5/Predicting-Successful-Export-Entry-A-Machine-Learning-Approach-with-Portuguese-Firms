import os, csv, sys

# sobe dois niveis: distance-to-export-python e paper_distanceToExport_v10_full_package
ROOTS = [".", ".."]
SKIP = {".git", "__pycache__", ".venv", "venv", "env", ".ipynb_checkpoints",
        "raw", "processed", "models", "predictions", "node_modules"}
seen = set()

print("=== 1. ONDE ESTAO AS PASTAS 'outputs' ===")
for R in ROOTS:
    for d, dirs, files in os.walk(R):
        dirs[:] = [x for x in dirs if x not in SKIP]
        if os.path.basename(d).lower() in ("outputs", "output", "results", "tables", "diagnostics"):
            n = len(files)
            print(f"{d}   ({n} ficheiros)")
            for f in sorted(files)[:40]:
                p = os.path.join(d, f)
                try: print(f"      {os.path.getsize(p)//1024:>7} KB  {f}")
                except OSError: pass

print("\n=== 2. FICHEIROS COM best_params, roc_auc OU 'chosen' ===")
META = ("method","model","model_name","sample","outcome","split_design","split",
        "feature_set","params","best_params","chosen","status","roc_auc","pr_auc",
        "lift_at_5","lift_at_10","precision_at_10","test_events","n_test")
for R in ROOTS:
    for d, dirs, files in os.walk(R):
        dirs[:] = [x for x in dirs if x not in SKIP]
        for f in sorted(files):
            if not f.lower().endswith((".csv", ".json")): continue
            p = os.path.realpath(os.path.join(d, f))
            if p in seen: continue
            seen.add(p)
            try:
                if os.path.getsize(p) > 20_000_000: continue
                txt = open(p, encoding="utf-8", errors="ignore").read()
            except OSError: continue
            if not any(k in txt for k in ("best_params", "roc_auc", '"chosen"', ",chosen")): continue
            print(f"\n--- {p}")
            if f.lower().endswith(".json"):
                print(txt[:4000]); continue
            rows = list(csv.DictReader(txt.splitlines()))
            if not rows: continue
            keep = [c for c in rows[0] if c in META] or list(rows[0])[:8]
            for r in rows[:80]:
                print(" | ".join(f"{c}={r.get(c,'')}" for c in keep))

print("\n=== 3. TABELAS LATEX JA GERADAS (para confirmar a pasta certa) ===")
for R in ROOTS:
    for d, dirs, files in os.walk(R):
        dirs[:] = [x for x in dirs if x not in SKIP]
        for f in files:
            if f.startswith("tab_") and f.endswith(".tex"):
                print(os.path.join(d, f))