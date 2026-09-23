# %%

"""02_leakage_identity_check.py
Test whether the old contemporaneous classifier was mechanically leaky.

We check:
  (1) Total_Vendas  ==  domestic + EU + extra-EU sales      (the accounting identity)
  (2) the old exporter label can be reconstructed from (Total_Vendas, MercadoInterno_Vendas)
      because exports = Total_Vendas - MercadoInterno_Vendas, and
      label = 1[ (Total_Vendas - MercadoInterno_Vendas) / Total_Vendas >= 0.10 ].

If (1) holds and (2) reproduces the label, any model given both Total_Vendas and
MercadoInterno_Vendas can recover the label within each row -> inflated AUC.

Output: outputs/tables/leakage_identity_check.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from _common import (load_config, load_raw_panel, standardise_ids, ensure_dirs,
                     resolve, to_numeric, safe_divide)


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    tables = resolve(cfg["paths"]["tables_dir"])
    C = cfg["columns"]
    thr = float(cfg["outcome"]["successful_export_intensity_threshold"])

    df = load_raw_panel(cfg)
    df = standardise_ids(df, cfg)
    df = to_numeric(df, [C["total_sales"], C["domestic_sales"], C["eu_sales"], C["extra_eu_sales"]])

    yr = C["year"]
    tv = df[C["total_sales"]].fillna(0.0)
    dom = df[C["domestic_sales"]].fillna(0.0)
    eu = df[C["eu_sales"]].fillna(0.0)
    xeu = df[C["extra_eu_sales"]].fillna(0.0)

    sales_sum = dom + eu + xeu
    diff = tv - sales_sum
    rel = pd.Series(safe_divide(diff.abs(), tv.abs()))

    # labels
    exports_true = eu + xeu
    old_label = pd.Series(np.where(tv > 0, (safe_divide(exports_true, tv) >= thr).astype(float), 0.0))
    recon_label = pd.Series(np.where(tv > 0, (safe_divide(tv - dom, tv) >= thr).astype(float), 0.0))
    same = (old_label.values == recon_label.values)

    g = pd.DataFrame({
        yr: df[yr].astype("Int64"),
        "abs_diff": diff.abs().values,
        "rel": rel.values,
        "below_1eur": (diff.abs() < 1).astype(int).values,
        "below_10eur": (diff.abs() < 10).astype(int).values,
        "below_0p1pct": (rel < 0.001).astype(int).values,
        "label_same": same.astype(int),
    })

    out = (g.groupby(yr)
             .agg(observations=("abs_diff", "size"),
                  mean_abs_diff=("abs_diff", "mean"),
                  median_abs_diff=("abs_diff", "median"),
                  p95_abs_diff=("abs_diff", lambda s: float(np.nanpercentile(s, 95))),
                  share_below_1eur=("below_1eur", "mean"),
                  share_below_10eur=("below_10eur", "mean"),
                  share_below_0p1pct=("below_0p1pct", "mean"),
                  share_label_reconstructed=("label_same", "mean"))
             .reset_index())

    # overall row
    overall = pd.DataFrame([{
        yr: "ALL",
        "observations": len(g),
        "mean_abs_diff": g["abs_diff"].mean(),
        "median_abs_diff": g["abs_diff"].median(),
        "p95_abs_diff": float(np.nanpercentile(g["abs_diff"], 95)),
        "share_below_1eur": g["below_1eur"].mean(),
        "share_below_10eur": g["below_10eur"].mean(),
        "share_below_0p1pct": g["below_0p1pct"].mean(),
        "share_label_reconstructed": g["label_same"].mean(),
    }])
    out = pd.concat([out, overall], ignore_index=True)
    out.to_csv(tables / "leakage_identity_check.csv", index=False)

    print("\n=== LEAKAGE IDENTITY CHECK ===")
    print(out.to_string(index=False))
    s_id = float(g["below_1eur"].mean())
    s_lab = float(g["label_same"].mean())
    print(f"\nIdentity holds to <1 euro in {s_id:6.2%} of firm-years.")
    print(f"Old label reproduced from (Total_Vendas, MercadoInterno_Vendas) in {s_lab:6.2%} of rows.")
    if s_id > 0.95 and s_lab > 0.95:
        print(">> Confirms within-row leakage in the old design. Do not rely on the old AUCs;\n"
              "   exclude Total_Vendas and the sales-decomposition variables from predictors.")
    else:
        print(">> Identity weaker than expected — still exclude sales-decomposition variables,\n"
              "   and inspect rows where it fails (services, reporting gaps).")
    print(f"\nWrote: {tables/'leakage_identity_check.csv'}")


if __name__ == "__main__":
    main()
