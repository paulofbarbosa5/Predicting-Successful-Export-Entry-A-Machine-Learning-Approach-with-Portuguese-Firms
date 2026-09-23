# %%

"""05_make_splits.py
Create leakage-aware modeling samples and split assignments.

Default headline design based on diagnostics:
    outcome = entry_from_zero_to_success_t1
    horizon = 1
    main time split candidate years:
        train: 2010-2017
        valid: 2018
        test : 2019-2020

Outputs:
    data/processed/model_sample_<outcome>.parquet
    outputs/tables/split_diagnostics_<outcome>.csv
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from _common import load_config, ensure_dirs, resolve
from _model_common import default_time_windows, horizon_for_outcome, slug


def assign_random_firm_year(df: pd.DataFrame, y: str, seed: int) -> pd.Series:
    idx = df.index.to_numpy()
    strat = df[y].astype(int) if df[y].nunique() == 2 else None
    train_idx, rest_idx = train_test_split(idx, test_size=0.30, random_state=seed, stratify=strat)
    rest = df.loc[rest_idx]
    strat_rest = rest[y].astype(int) if rest[y].nunique() == 2 else None
    valid_idx, test_idx = train_test_split(rest_idx, test_size=2/3, random_state=seed + 1, stratify=strat_rest)
    out = pd.Series(pd.NA, index=df.index, dtype="string")
    out.loc[train_idx] = "train"
    out.loc[valid_idx] = "valid"
    out.loc[test_idx] = "test"
    return out


def assign_firm_grouped(df: pd.DataFrame, fid: str, y: str, seed: int) -> pd.Series:
    firms = df[[fid]].drop_duplicates()[fid].to_numpy()
    # Stratify at firm level by whether the firm ever enters in this modeling sample.
    ever = df.groupby(fid)[y].max().reindex(firms).fillna(0).astype(int).to_numpy()
    strat = ever if len(np.unique(ever)) == 2 else None
    train_firms, rest_firms = train_test_split(firms, test_size=0.30, random_state=seed, stratify=strat)
    rest_ever = df.groupby(fid)[y].max().reindex(rest_firms).fillna(0).astype(int).to_numpy()
    strat_rest = rest_ever if len(np.unique(rest_ever)) == 2 else None
    valid_firms, test_firms = train_test_split(rest_firms, test_size=2/3, random_state=seed + 1, stratify=strat_rest)
    out = pd.Series(pd.NA, index=df.index, dtype="string")
    out.loc[df[fid].isin(train_firms)] = "train"
    out.loc[df[fid].isin(valid_firms)] = "valid"
    out.loc[df[fid].isin(test_firms)] = "test"
    return out


def assign_time(df: pd.DataFrame, yr: str, windows: dict[str, list[int]]) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="string")
    for split, years in windows.items():
        out.loc[df[yr].isin(years)] = split
    return out


def assign_firm_grouped_time(df: pd.DataFrame, fid: str, yr: str, windows: dict[str, list[int]], y: str, seed: int) -> pd.Series:
    firms = df[fid].drop_duplicates().to_numpy()
    ever = df.groupby(fid)[y].max().reindex(firms).fillna(0).astype(int).to_numpy()
    strat = ever if len(np.unique(ever)) == 2 else None
    train_firms, rest_firms = train_test_split(firms, test_size=0.30, random_state=seed, stratify=strat)
    rest_ever = df.groupby(fid)[y].max().reindex(rest_firms).fillna(0).astype(int).to_numpy()
    strat_rest = rest_ever if len(np.unique(rest_ever)) == 2 else None
    valid_firms, test_firms = train_test_split(rest_firms, test_size=2/3, random_state=seed + 1, stratify=strat_rest)
    out = pd.Series(pd.NA, index=df.index, dtype="string")
    out.loc[df[fid].isin(train_firms) & df[yr].isin(windows["train"])] = "train"
    out.loc[df[fid].isin(valid_firms) & df[yr].isin(windows["valid"])] = "valid"
    out.loc[df[fid].isin(test_firms) & df[yr].isin(windows["test"])] = "test"
    return out


def diagnose_split(df: pd.DataFrame, fid: str, yr: str, y: str, col: str) -> dict[str, object]:
    row: dict[str, object] = {"split_design": col.replace("split_", "")}
    for part in ["train", "valid", "test"]:
        d = df[df[col] == part]
        row[f"{part}_n"] = len(d)
        row[f"{part}_firms"] = d[fid].nunique()
        row[f"{part}_events"] = int(d[y].sum()) if len(d) else 0
        row[f"{part}_entry_rate"] = float(d[y].mean()) if len(d) else np.nan
        row[f"{part}_years"] = ",".join(map(str, sorted(d[yr].dropna().unique())))
    train_firms = set(df.loc[df[col] == "train", fid])
    test_firms = set(df.loc[df[col] == "test", fid])
    train_years = set(df.loc[df[col] == "train", yr])
    test_years = set(df.loc[df[col] == "test", yr])
    row["firm_overlap_train_test"] = len(train_firms & test_firms)
    row["year_overlap_train_test"] = len(train_years & test_years)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default="entry_from_zero_to_success_t1")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--include-robustness-splits", action="store_true", help="Add pre_covid_time and post2018_time when valid.")
    args, _ = ap.parse_known_args()

    cfg = load_config()
    ensure_dirs(cfg)
    fid, yr = cfg["columns"]["firm_id"], cfg["columns"]["year"]
    processed = resolve(cfg["paths"]["processed_dir"])
    tables = resolve(cfg["paths"]["tables_dir"])
    panel_path = processed / "entry_panel.parquet"
    if not panel_path.exists():
        sys.exit(f"{panel_path} not found. Run scripts 01–04 first.")
    df = pd.read_parquet(panel_path)
    if args.outcome not in df.columns:
        sys.exit(f"Outcome {args.outcome} not found in entry_panel.parquet")

    h = horizon_for_outcome(args.outcome)
    max_valid_year = int(df[yr].max()) - h
    sample = df[df[args.outcome].notna() & (df[yr] <= max_valid_year)].copy()
    sample[args.outcome] = sample[args.outcome].astype(int)
    sample = sample.reset_index(drop=True)

    if sample[args.outcome].nunique() < 2:
        sys.exit("Selected outcome has only one class after filtering. Choose another outcome.")

    min_y, max_y = int(sample[yr].min()), int(df[yr].max())
    main_windows = default_time_windows(min_y, max_y, h, design="main")
    print(f"[splits] outcome={args.outcome} horizon={h}")
    print(f"[splits] main candidate-year windows: {main_windows}")

    sample["split_random_firm_year"] = assign_random_firm_year(sample, args.outcome, args.seed)
    sample["split_firm_grouped"] = assign_firm_grouped(sample, fid, args.outcome, args.seed)
    sample["split_time"] = assign_time(sample, yr, main_windows)
    sample["split_firm_grouped_time"] = assign_firm_grouped_time(sample, fid, yr, main_windows, args.outcome, args.seed)

    split_cols = ["split_random_firm_year", "split_firm_grouped", "split_time", "split_firm_grouped_time"]

    if args.include_robustness_splits:
        try:
            pre = default_time_windows(min_y, max_y, h, design="pre_covid")
            sample["split_pre_covid_time"] = assign_time(sample, yr, pre)
            split_cols.append("split_pre_covid_time")
            print(f"[splits] pre-COVID candidate-year windows: {pre}")
        except Exception as e:
            print(f"[splits] skipping pre-COVID split: {e}")
        try:
            post = default_time_windows(min_y, max_y, h, design="post2018")
            sample["split_post2018_time"] = assign_time(sample, yr, post)
            split_cols.append("split_post2018_time")
            print(f"[splits] post-2018 candidate-year windows: {post}")
        except Exception as e:
            print(f"[splits] skipping post-2018 split: {e}")

    diag = pd.DataFrame([diagnose_split(sample, fid, yr, args.outcome, c) for c in split_cols])
    diag_path = tables / f"split_diagnostics_{slug(args.outcome)}.csv"
    diag.to_csv(diag_path, index=False)

    out_path = processed / f"model_sample_{slug(args.outcome)}.parquet"
    sample.to_parquet(out_path, index=False)

    print("\n=== SPLIT DIAGNOSTICS ===")
    print(diag.to_string(index=False))
    print(f"\nSaved {out_path}")
    print(f"Saved {diag_path}")


if __name__ == "__main__":
    main()
