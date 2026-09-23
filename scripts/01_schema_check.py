# %%

"""01_schema_check.py
Confirm Python can read iesCompleto.RData and that the variables we need exist.
Outputs:
    outputs/tables/schema_columns.csv
    outputs/tables/raw_year_counts.csv
    outputs/tables/raw_missing_summary.csv
This is read-only. It does not build anything.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from _common import (load_config, load_raw_panel, standardise_ids, ensure_dirs,
                     resolve, derive_division)


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    tables = resolve(cfg["paths"]["tables_dir"])

    df = load_raw_panel(cfg)
    df = standardise_ids(df, cfg)
    fid, yr = cfg["columns"]["firm_id"], cfg["columns"]["year"]

    # ---- column presence -------------------------------------------------
    needed = {k: v for k, v in cfg["columns"].items()}
    rows = []
    for role, col in needed.items():
        present = col in df.columns
        rows.append({"role": role, "column": col, "present": present,
                     "dtype": str(df[col].dtype) if present else ""})
    schema = pd.DataFrame(rows)
    schema.to_csv(tables / "schema_columns.csv", index=False)

    missing_cols = schema.loc[~schema["present"], "column"].tolist()
    print("\n=== COLUMN PRESENCE ===")
    print(schema.to_string(index=False))
    if missing_cols:
        print(f"\n!!! MISSING {len(missing_cols)} expected column(s): {missing_cols}")
        print("    Fix the names in config/config.yaml before running later scripts.")
    else:
        print("\nAll expected columns are present.")

    # ---- duplicate firm-years -------------------------------------------
    dup = int(df.duplicated(subset=[fid, yr]).sum())
    print(f"\n=== STRUCTURE ===\nrows={len(df):,}  unique firms={df[fid].nunique():,}  "
          f"duplicate {fid}-{yr} rows={dup:,}")
    if dup:
        print("    NOTE duplicates will be dropped (keep first) in script 03.")

    # ---- yearly counts (the 2018 jump should show here) ------------------
    yc = (df.groupby(yr)
            .agg(firm_years=(fid, "size"), unique_firms=(fid, "nunique"))
            .reset_index())
    yc.to_csv(tables / "raw_year_counts.csv", index=False)
    print("\n=== FIRM-YEARS BY YEAR (watch for the ~2018 jump) ===")
    print(yc.to_string(index=False))

    # ---- missingness -----------------------------------------------------
    miss = (df.isna().mean().sort_values(ascending=False)
              .rename("missing_rate").reset_index()
              .rename(columns={"index": "column"}))
    miss.to_csv(tables / "raw_missing_summary.csv", index=False)

    # ---- sanity on key magnitudes + sector derivation preview ------------
    sales = cfg["columns"]["total_sales"]
    emp = cfg["columns"]["employment"]
    assets = cfg["columns"]["assets"]
    print("\n=== SANITY CHECKS ===")
    for col in (sales, emp, assets):
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            print(f"{col}: nonpositive(<=0)={int((x <= 0).sum()):,}  "
                  f"missing={int(x.isna().sum()):,}  median={np.nanmedian(x):,.0f}")

    sec = cfg["columns"]["sector_raw"]
    if sec in df.columns:
        div = derive_division(df[sec])
        man = set(cfg["panel"]["manufacturing_divisions"])
        in_man = div.isin(man).sum()
        print(f"\nDerived 2-digit division from {sec}: "
              f"{int(in_man):,} firm-years in manufacturing divisions {sorted(man)}")
        print("Sample CAE.x -> division:")
        print(pd.DataFrame({sec: df[sec].head(8).values, "division": div.head(8).values})
              .to_string(index=False))

    print(f"\nWrote: {tables/'schema_columns.csv'}, raw_year_counts.csv, raw_missing_summary.csv")


if __name__ == "__main__":
    main()
