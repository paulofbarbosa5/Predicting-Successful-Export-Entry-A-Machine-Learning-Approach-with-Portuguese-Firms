# %%

"""01b_duplicate_diagnostics.py
Inspect duplicate firm-year rows before final modeling.

This matters because the diagnostic outputs showed many duplicate NIPC-Ano rows in 2021.
Script 03 currently keeps the first duplicate. If duplicates differ materially, you should
aggregate or resolve them before building the panel.

Outputs:
    outputs/tables/duplicate_firm_year_summary.csv
    outputs/tables/duplicate_firm_year_differences.csv
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from _common import load_config, load_raw_panel, standardise_ids, ensure_dirs, resolve, to_numeric


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    C = cfg["columns"]
    fid, yr = C["firm_id"], C["year"]
    tables = resolve(cfg["paths"]["tables_dir"])

    df = load_raw_panel(cfg)
    df = standardise_ids(df, cfg)
    df = df.dropna(subset=[fid, yr]).copy()
    df[yr] = df[yr].astype(int)

    check_cols = [
        C["sector_raw"], C["domestic_sales"], C["eu_sales"], C["extra_eu_sales"], C["total_sales"],
        C["total_purchases"], C["material_costs_total"], C["assets"], C["equity"],
        C["employment"], C["net_income"], C["personnel_costs_total"],
    ]
    numeric_cols = [c for c in check_cols if c != C["sector_raw"]]
    df = to_numeric(df, numeric_cols)

    grp = df.groupby([fid, yr], dropna=False)
    size = grp.size().rename("n_rows").reset_index()
    dup_keys = size[size["n_rows"] > 1]

    summary = (
        size.groupby(yr)
        .agg(
            firm_year_keys=(fid, "size"),
            duplicate_keys=("n_rows", lambda s: int((s > 1).sum())),
            extra_rows=("n_rows", lambda s: int((s - 1).clip(lower=0).sum())),
            max_rows_per_key=("n_rows", "max"),
        )
        .reset_index()
    )
    summary.to_csv(tables / "duplicate_firm_year_summary.csv", index=False)

    diff_rows = []
    if len(dup_keys):
        dup_df = df.merge(dup_keys[[fid, yr]], on=[fid, yr], how="inner")
        for (year, col), dd in dup_df.groupby([yr, C["sector_raw"]], dropna=False):
            pass  # placeholder to keep grouping import-free

        for col in check_cols:
            if col not in dup_df.columns:
                continue
            if col in numeric_cols:
                stat = grp[col].agg(lambda s: float(np.nanmax(s) - np.nanmin(s)) if len(s) else np.nan).rename("range")
                tmp = stat.reset_index().merge(dup_keys[[fid, yr]], on=[fid, yr], how="inner")
                diff_rows.append({
                    "column": col,
                    "duplicate_keys": len(tmp),
                    "share_different": float((tmp["range"].fillna(0) != 0).mean()) if len(tmp) else np.nan,
                    "mean_abs_range": float(tmp["range"].abs().mean()) if len(tmp) else np.nan,
                    "p95_abs_range": float(tmp["range"].abs().quantile(0.95)) if len(tmp) else np.nan,
                })
            else:
                stat = grp[col].nunique(dropna=False).rename("n_unique_values")
                tmp = stat.reset_index().merge(dup_keys[[fid, yr]], on=[fid, yr], how="inner")
                diff_rows.append({
                    "column": col,
                    "duplicate_keys": len(tmp),
                    "share_different": float((tmp["n_unique_values"] > 1).mean()) if len(tmp) else np.nan,
                    "mean_abs_range": np.nan,
                    "p95_abs_range": np.nan,
                })

    pd.DataFrame(diff_rows).to_csv(tables / "duplicate_firm_year_differences.csv", index=False)

    print("\n=== DUPLICATE FIRM-YEAR SUMMARY ===")
    print(summary.to_string(index=False))
    print("\nIf duplicate values differ materially, resolve duplicates before final modeling.")
    print(f"Tables written to {tables}")


if __name__ == "__main__":
    main()
