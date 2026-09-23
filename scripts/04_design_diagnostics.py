# %%

"""04_design_diagnostics.py
Gatekeeper diagnostics. Decide the headline outcome, horizon, size cells, and split
from the data BEFORE running models.

Outputs:
    outputs/tables/event_counts_by_outcome.csv
    outputs/tables/event_counts_by_year.csv
    outputs/tables/event_counts_by_size.csv
    outputs/tables/valid_candidate_years_by_horizon.csv
    outputs/tables/pre_post_2018_composition.csv
    outputs/tables/pre_post_2018_entry_rates.csv
    outputs/tables/design_recommendation.csv
    outputs/tables/event_counts_after_feature_requirements.csv
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from _common import load_config, ensure_dirs, resolve

OUTCOME_HORIZON = {
    "entry_from_zero_to_positive_t1": 1,
    "entry_from_zero_to_success_t1": 1,
    "entry_from_zero_to_success_t1_t2": 2,
    "entry_from_zero_to_success_t1_t3": 3,
    "entry_from_near_zero_to_success_t1": 1,
    "entry_from_near_zero_to_success_t1_t2": 2,
    "transition_below10_to_success_t1": 1,
    "transition_below10_to_success_t1_t2": 2,
    "transition_positive_below10_to_success_t1": 1,
    "transition_positive_below10_to_success_t1_t2": 2,
}

FEATURE_BASES = [
    "labor_productivity_vab",
    "imports_comu",
    "imports_extra",
    "import_experience",
    "log_employment",
    "log_assets",
    "capital_intensity",
    "wage_per_worker",
    "equity_ratio",
    "roa",
    "profit_margin",
    "negative_equity",
]
FEATURE_SETS = {
    "main_t": FEATURE_BASES + ["division", "size_class"],
    "main_l1": [f"L1_{b}" for b in FEATURE_BASES] + ["division", "size_class"],
    "no_sales_derived_t": [b for b in FEATURE_BASES if b not in {"labor_productivity_vab", "profit_margin"}] + ["division", "size_class"],
    "no_sales_derived_l1": [f"L1_{b}" for b in FEATURE_BASES if b not in {"labor_productivity_vab", "profit_margin"}] + ["division", "size_class"],
    "no_imports_t": [b for b in FEATURE_BASES if b not in {"imports_comu", "imports_extra", "import_experience"}] + ["division", "size_class"],
    "no_imports_l1": [f"L1_{b}" for b in FEATURE_BASES if b not in {"imports_comu", "imports_extra", "import_experience"}] + ["division", "size_class"],
}

MIN_HEADLINE_TEST_EVENTS = 100
MIN_MICRO_SMALL_EVENTS = 50
MIN_ROBUSTNESS_TEST_EVENTS = 50
MIN_APPENDIX_TEST_EVENTS = 20


def test_window(max_year: int, h: int) -> list[int]:
    """Last two valid candidate years for horizon h."""
    last_valid = max_year - h
    return [last_valid - 1, last_valid] if last_valid - 1 >= 0 else [last_valid]


def feature_complete_mask(df: pd.DataFrame, features: list[str]) -> pd.Series:
    """Rows with observed target features. Categorical division/size must be nonmissing;
    numeric values may still be imputed in modeling, but this diagnostic shows complete cases.
    """
    present = [c for c in features if c in df.columns]
    if len(present) != len(features):
        missing = sorted(set(features) - set(present))
        print(f"[diagnostics] WARNING missing feature columns in panel: {missing}")
    return df[present].notna().all(axis=1)


def classify_use(test_events: int, micro_small_events: int) -> str:
    if test_events >= MIN_HEADLINE_TEST_EVENTS and micro_small_events >= MIN_MICRO_SMALL_EVENTS:
        return "headline candidate"
    if test_events >= MIN_ROBUSTNESS_TEST_EVENTS:
        return "main robustness"
    if test_events >= MIN_APPENDIX_TEST_EVENTS:
        return "appendix only"
    return "too sparse"


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    tables = resolve(cfg["paths"]["tables_dir"])
    fid, yr = cfg["columns"]["firm_id"], cfg["columns"]["year"]

    panel_path = resolve(cfg["paths"]["processed_dir"]) / "entry_panel.parquet"
    if not panel_path.exists():
        sys.exit(f"{panel_path} not found. Run scripts 01–03 first.")
    df = pd.read_parquet(panel_path)
    min_y, max_y = int(df[yr].min()), int(df[yr].max())
    outcomes = [o for o in OUTCOME_HORIZON if o in df.columns]

    # Overall counts per outcome.
    rows = []
    for o in outcomes:
        d = df[df[o].notna()]
        rows.append({
            "outcome": o,
            "horizon": OUTCOME_HORIZON[o],
            "candidate_firm_years": len(d),
            "unique_firms": d[fid].nunique(),
            "entry_events": int(d[o].sum()),
            "entry_rate": float(d[o].mean()) if len(d) else np.nan,
            "first_candidate_year": int(d[yr].min()) if len(d) else None,
            "last_candidate_year": int(d[yr].max()) if len(d) else None,
        })
    by_outcome = pd.DataFrame(rows)
    by_outcome.to_csv(tables / "event_counts_by_outcome.csv", index=False)

    # By year.
    yr_rows = []
    for o in outcomes:
        d = df[df[o].notna()]
        if not len(d):
            continue
        t = d.groupby(yr).agg(candidates=(o, "size"), events=(o, "sum"), rate=(o, "mean")).reset_index()
        t.insert(0, "outcome", o)
        yr_rows.append(t)
    if yr_rows:
        pd.concat(yr_rows, ignore_index=True).to_csv(tables / "event_counts_by_year.csv", index=False)

    # By size.
    sz_rows = []
    for o in outcomes:
        d = df[df[o].notna()]
        if not len(d):
            continue
        t = d.groupby("size_class", dropna=False).agg(candidates=(o, "size"), events=(o, "sum"), rate=(o, "mean")).reset_index()
        t.insert(0, "outcome", o)
        sz_rows.append(t)
    if sz_rows:
        pd.concat(sz_rows, ignore_index=True).to_csv(tables / "event_counts_by_size.csv", index=False)

    # Valid candidate years by horizon.
    vy = []
    for h in (1, 2, 3):
        last_valid = max_y - h
        vy.append({
            "horizon": h,
            "data_min_year": min_y,
            "data_max_year": max_y,
            "max_valid_candidate_year": last_valid,
            "n_valid_candidate_years": max(0, last_valid - min_y + 1),
            "suggested_test_candidate_years": str(test_window(max_y, h)),
        })
    pd.DataFrame(vy).to_csv(tables / "valid_candidate_years_by_horizon.csv", index=False)

    # Pre/post-2018 composition.
    df["_period"] = np.where(df[yr] <= 2017, "2010-2017", "2018-2021")
    comp = (
        df.groupby("_period")
        .agg(
            firm_years=(fid, "size"),
            unique_firms=(fid, "nunique"),
            share_micro=("size_class", lambda s: (s == "micro").mean()),
            share_small=("size_class", lambda s: (s == "small").mean()),
            share_medium=("size_class", lambda s: (s == "medium").mean()),
            share_large=("size_class", lambda s: (s == "large").mean()),
            share_exporter10=("exporter_10", "mean"),
            share_zero_exporter=("candidate_zero", "mean"),
        )
        .reset_index()
    )
    comp.to_csv(tables / "pre_post_2018_composition.csv", index=False)

    # Entry rates pre/post by outcome.
    er_rows = []
    for o in outcomes:
        d = df[df[o].notna()]
        for per, dd in d.groupby("_period"):
            ent = dd[dd[o] == 1]
            er_rows.append({
                "outcome": o,
                "period": per,
                "candidates": len(dd),
                "events": int(dd[o].sum()),
                "entry_rate": float(dd[o].mean()) if len(dd) else np.nan,
                "micro_share_of_entrants": float((ent["size_class"] == "micro").mean()) if len(ent) else np.nan,
            })
    pd.DataFrame(er_rows).to_csv(tables / "pre_post_2018_entry_rates.csv", index=False)

    # Recommendation table.
    rec_rows = []
    for o in outcomes:
        h = OUTCOME_HORIZON[o]
        win = test_window(max_y, h)
        d = df[df[o].notna()]
        dt = d[d[yr].isin(win)]
        te = int(dt[o].sum())
        by_sz = dt.groupby("size_class")[o].sum()
        ms = int(by_sz.get("micro", 0) + by_sz.get("small", 0))
        rec_rows.append({
            "outcome": o,
            "horizon": h,
            "test_candidate_years": str(win),
            "test_events": te,
            "micro_events": int(by_sz.get("micro", 0)),
            "small_events": int(by_sz.get("small", 0)),
            "medium_events": int(by_sz.get("medium", 0)),
            "large_events": int(by_sz.get("large", 0)),
            "recommended_use": classify_use(te, ms),
        })
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(tables / "design_recommendation.csv", index=False)

    # Feature-complete diagnostics.
    fc_rows = []
    for o in outcomes:
        h = OUTCOME_HORIZON[o]
        win = test_window(max_y, h)
        d0 = df[df[o].notna()].copy()
        for fs_name, features in FEATURE_SETS.items():
            m = feature_complete_mask(d0, features)
            d = d0[m]
            dt = d[d[yr].isin(win)]
            by_sz = dt.groupby("size_class")[o].sum()
            fc_rows.append({
                "outcome": o,
                "horizon": h,
                "feature_set": fs_name,
                "candidate_firm_years_complete": len(d),
                "entry_events_complete": int(d[o].sum()) if len(d) else 0,
                "entry_rate_complete": float(d[o].mean()) if len(d) else np.nan,
                "test_candidate_years": str(win),
                "test_events_complete": int(dt[o].sum()) if len(dt) else 0,
                "micro_test_events_complete": int(by_sz.get("micro", 0)),
                "small_test_events_complete": int(by_sz.get("small", 0)),
                "medium_test_events_complete": int(by_sz.get("medium", 0)),
                "large_test_events_complete": int(by_sz.get("large", 0)),
            })
    pd.DataFrame(fc_rows).to_csv(tables / "event_counts_after_feature_requirements.csv", index=False)

    print("\n=== EVENT COUNTS BY OUTCOME ===")
    print(by_outcome.to_string(index=False))
    print("\n=== PRE / POST 2018 COMPOSITION ===")
    print(comp.to_string(index=False))
    print("\n=== DESIGN RECOMMENDATION (test window = last two valid candidate years) ===")
    print(rec.to_string(index=False))
    print("\nTables written to", tables)
    print("\nNext: choose selected_outcome, selected_horizon, selected_feature_set, and run script 05.")


if __name__ == "__main__":
    main()
