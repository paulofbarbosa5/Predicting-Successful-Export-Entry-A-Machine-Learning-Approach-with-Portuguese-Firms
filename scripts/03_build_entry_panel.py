# %%

"""03_build_entry_panel.py
Build the forward-looking firm-year panel that replaces data_creationV2.R.

Key choices:
  * Productivity = VAB / employment, VAB = Total_Vendas - CustoMercadoriasMateriasConsumidas_Total.
  * capital_intensity = TotalActivo / employment (NOT assets / sales).
  * EU import share = (EU Compras + EU Fornecimentos) / Total_Compras.
  * Sector = 2-digit Divisao derived from CAE.x; manufacturing only.
  * Candidate families:
        - zero exporters: true export entry;
        - near-zero exporters: robustness for tiny/rounding exports;
        - below-10% firms: successful-export transition.
  * Leads/lags are valid only when neighbouring years are exactly +/- the horizon.
  * Multi-year outcomes require a full future window. For example, t+1:t+2 is missing
    when t+2 is not observed.
  * Sales-decomposition variables are kept ONLY to build outcomes/productivity, never as features.

Output:
    data/processed/entry_panel.parquet
"""
from __future__ import annotations

import os
import sys
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from _common import (
    load_config,
    load_raw_panel,
    standardise_ids,
    ensure_dirs,
    resolve,
    to_numeric,
    safe_divide,
    derive_division,
)

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


def window_event_full(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Return max over lead status columns, but only if every lead column is observed.

    This avoids coding 2020 as a valid t+1:t+2 non-entry when 2022 is unobserved.
    """
    arr = df[cols].to_numpy(dtype="float64")
    full_window = np.all(~np.isnan(arr), axis=1)
    out = np.full(arr.shape[0], np.nan, dtype="float64")
    if full_window.any():
        out[full_window] = np.nanmax(arr[full_window], axis=1)
    return out


def log_positive(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(x, dtype="float64")
    return np.where(arr > 0, np.log(arr), np.nan)


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    C = cfg["columns"]
    thr = float(cfg["outcome"]["successful_export_intensity_threshold"])
    near_thr = float(cfg["outcome"].get("near_zero_export_intensity_threshold", 0.001))
    man = set(cfg["panel"]["manufacturing_divisions"])
    sb = cfg["panel"]["size_bins"]

    df = load_raw_panel(cfg)
    df = standardise_ids(df, cfg)
    fid, yr = C["firm_id"], C["year"]

    num_cols = [
        C[k]
        for k in (
            "total_sales",
            "domestic_sales",
            "eu_sales",
            "extra_eu_sales",
            "total_services",
            "total_purchases",
            "eu_purchases",
            "extra_eu_purchases",
            "eu_supplies",
            "extra_eu_supplies",
            "material_costs_total",
            "revenue",
            "net_income",
            "assets",
            "equity",
            "employment",
            "personnel_costs_total",
        )
    ]
    df = to_numeric(df, num_cols)

    # Drop rows without usable id/year and exact duplicate firm-years.
    # If duplicates are not exact, inspect them with the duplicate diagnostic before final runs.
    df = df.dropna(subset=[fid, yr]).copy()
    df[yr] = df[yr].astype(int)
    before = len(df)
    df = df.drop_duplicates(subset=[fid, yr], keep="first")
    if len(df) != before:
        print(f"[build] dropped {before - len(df):,} duplicate {fid}-{yr} rows (kept first)")

    # Sector: 2-digit division, manufacturing only.
    df["division"] = derive_division(df[C["sector_raw"]])
    n0 = len(df)
    df = df[df["division"].isin(man)].copy()
    print(
        f"[build] manufacturing filter: kept {len(df):,} / {n0:,} firm-years "
        f"({df[fid].nunique():,} firms)"
    )

    # Size class from employment (this is the class used as a MODEL FEATURE).
    emp = df[C["employment"]]
    cond = [
        emp < sb["micro_max_employees"],
        (emp >= sb["micro_max_employees"]) & (emp < sb["small_max_employees"]),
        (emp >= sb["small_max_employees"]) & (emp < sb["medium_max_employees"]),
        emp >= sb["medium_max_employees"],
    ]
    df["size_class"] = np.select(cond, ["micro", "small", "medium", "large"], default=None)

    # EC Recommendation 2003/361 size class (turnover-only variant), DESCRIPTIVE ONLY.
    # Never used as a model feature; used only to check the robustness of the size
    # composition of entrants to the size definition. A firm is placed in the smallest
    # band whose headcount AND turnover ceilings are both satisfied; otherwise it is large.
    sb_ec = cfg["panel"].get("size_bins_ec", None)
    if sb_ec is not None:
        turnover_ec = df[C["total_sales"]].fillna(0.0) + df[C["total_services"]].fillna(0.0)
        assets_ec = df[C["assets"]].fillna(np.inf)  # missing assets -> cannot satisfy the OR via assets
        df["turnover_ec"] = turnover_ec.to_numpy()
        emp_known = emp.notna()

        def fin_ok(turn_ceiling, asset_ceiling):
            # EC rule: a band's financial test passes if turnover OR balance-sheet is within ceiling.
            ok = turnover_ec <= turn_ceiling
            if asset_ceiling is not None:
                ok = ok | (assets_ec <= asset_ceiling)
            return ok

        cond_ec = [
            (emp < sb["micro_max_employees"]) & fin_ok(sb_ec["micro_max_turnover_eur"], sb_ec.get("micro_max_assets_eur")),
            (emp < sb["small_max_employees"]) & fin_ok(sb_ec["small_max_turnover_eur"], sb_ec.get("small_max_assets_eur")),
            (emp < sb["medium_max_employees"]) & fin_ok(sb_ec["medium_max_turnover_eur"], sb_ec.get("medium_max_assets_eur")),
        ]
        # default 'large' for firms exceeding every SME band; None where employment is missing.
        df["size_class_ec"] = np.select(cond_ec, ["micro", "small", "medium"], default="large")
        df.loc[~emp_known, "size_class_ec"] = None
    else:
        df["size_class_ec"] = None

    # Exports and exporter status.
    eu = df[C["eu_sales"]].fillna(0.0)
    xeu = df[C["extra_eu_sales"]].fillna(0.0)
    tv = df[C["total_sales"]]
    df["exports_goods"] = (eu + xeu).to_numpy()
    inten = pd.Series(safe_divide(df["exports_goods"], tv), index=df.index)
    inten = inten.where(df["exports_goods"] > 0, 0.0)  # no exports -> 0% by definition
    df["export_intensity"] = inten.clip(lower=0.0, upper=1.0)
    df["exporter_positive"] = (df["exports_goods"] > 0).astype(int)
    df["exporter_10"] = (df["export_intensity"] >= thr).astype(int)

    # Productivity / scale.
    vab = tv - df[C["material_costs_total"]]
    df["labor_productivity_vab"] = safe_divide(vab, emp)
    df["capital_intensity"] = safe_divide(df[C["assets"]], emp)
    df["wage_per_worker"] = safe_divide(df[C["personnel_costs_total"]], emp)
    df["log_employment"] = log_positive(emp)
    df["log_assets"] = log_positive(df[C["assets"]])

    # Import experience / intensity.
    euc = df[C["eu_purchases"]].fillna(0.0)
    xeuc = df[C["extra_eu_purchases"]].fillna(0.0)
    eus = df[C["eu_supplies"]].fillna(0.0)
    xeus = df[C["extra_eu_supplies"]].fillna(0.0)
    tc = df[C["total_purchases"]]
    df["imports_comu"] = safe_divide(euc + eus, tc)
    df["imports_extra"] = safe_divide(xeuc + xeus, tc)
    df["import_experience"] = ((euc + xeuc + eus + xeus) > 0).astype(int)

    # Financial capacity.
    df["equity_ratio"] = safe_divide(df[C["equity"]], df[C["assets"]])
    # book_leverage = 1 - equity_ratio; computed for diagnostics only,
    # but excluded from default feature sets to avoid perfect collinearity.
    df["book_leverage"] = safe_divide(df[C["assets"]] - df[C["equity"]], df[C["assets"]])
    df["roa"] = safe_divide(df[C["net_income"]], df[C["assets"]])
    df["profit_margin"] = safe_divide(df[C["net_income"]], df[C["revenue"]])
    df["negative_equity"] = (df[C["equity"]] < 0).astype(int)

    # Candidate flags at t.
    df["candidate_zero"] = df["exports_goods"] == 0
    df["candidate_near_zero"] = df["export_intensity"] <= near_thr
    df["candidate_below10"] = df["export_intensity"] < thr
    df["candidate_positive_below10"] = (df["exports_goods"] > 0) & (df["export_intensity"] < thr)

    # Leads and lags with consecutive-year validity.
    df = df.sort_values([fid, yr]).reset_index(drop=True)
    g = df.groupby(fid, sort=False)

    for h in (1, 2, 3):
        ly = g[yr].shift(-h)
        valid = ly.eq(df[yr] + h)
        for src, dst in (("exporter_10", f"_lead10_{h}"), ("exporter_positive", f"_leadpos_{h}")):
            s = g[src].shift(-h)
            df[dst] = s.where(valid, np.nan)

    prev_valid = g[yr].shift(1).eq(df[yr] - 1)
    for base in FEATURE_BASES:
        df[f"L1_{base}"] = g[base].shift(1).where(prev_valid, np.nan)

    # Outcome family A: true export entry (candidate = zero exporter).
    cz = df["candidate_zero"].to_numpy()
    df["entry_from_zero_to_positive_t1"] = np.where(cz, df["_leadpos_1"], np.nan)
    df["entry_from_zero_to_success_t1"] = np.where(cz, df["_lead10_1"], np.nan)
    df["entry_from_zero_to_success_t1_t2"] = np.where(
        cz, window_event_full(df, ["_lead10_1", "_lead10_2"]), np.nan
    )
    df["entry_from_zero_to_success_t1_t3"] = np.where(
        cz, window_event_full(df, ["_lead10_1", "_lead10_2", "_lead10_3"]), np.nan
    )

    # Near-zero robustness family.
    cnz = df["candidate_near_zero"].to_numpy()
    df["entry_from_near_zero_to_success_t1"] = np.where(cnz, df["_lead10_1"], np.nan)
    df["entry_from_near_zero_to_success_t1_t2"] = np.where(
        cnz, window_event_full(df, ["_lead10_1", "_lead10_2"]), np.nan
    )

    # Outcome family B: successful-export transition (candidate = below 10%).
    cb = df["candidate_below10"].to_numpy()
    df["transition_below10_to_success_t1"] = np.where(cb, df["_lead10_1"], np.nan)
    df["transition_below10_to_success_t1_t2"] = np.where(
        cb, window_event_full(df, ["_lead10_1", "_lead10_2"]), np.nan
    )

    # Pure threshold-crossing robustness: positive exporter below 10% -> successful exporter.
    cpb = df["candidate_positive_below10"].to_numpy()
    df["transition_positive_below10_to_success_t1"] = np.where(cpb, df["_lead10_1"], np.nan)
    df["transition_positive_below10_to_success_t1_t2"] = np.where(
        cpb, window_event_full(df, ["_lead10_1", "_lead10_2"]), np.nan
    )

    outcome_cols = [
        "entry_from_zero_to_positive_t1",
        "entry_from_zero_to_success_t1",
        "entry_from_zero_to_success_t1_t2",
        "entry_from_zero_to_success_t1_t3",
        "entry_from_near_zero_to_success_t1",
        "entry_from_near_zero_to_success_t1_t2",
        "transition_below10_to_success_t1",
        "transition_below10_to_success_t1_t2",
        "transition_positive_below10_to_success_t1",
        "transition_positive_below10_to_success_t1_t2",
    ]
    feature_cols = [f"L1_{b}" for b in FEATURE_BASES] + FEATURE_BASES + ["division", "size_class"]
    # Descriptive (non-feature) columns carried through for robustness checks.
    descriptive_cols = [c for c in ["size_class_ec", "turnover_ec"] if c in df.columns]
    construction_cols = [
        "exports_goods",
        "export_intensity",
        "exporter_10",
        "exporter_positive",
        "candidate_zero",
        "candidate_near_zero",
        "candidate_below10",
        "candidate_positive_below10",
    ]
    keep = [fid, yr] + construction_cols + outcome_cols + feature_cols + descriptive_cols
    panel = df[keep].copy()

    out_path = resolve(cfg["paths"]["processed_dir"]) / "entry_panel.parquet"
    panel.to_parquet(out_path, index=False)

    print(f"\n[build] saved {out_path}  shape={panel.shape}")
    print("\nMODEL FEATURES (use L1_* for the conservative design):")
    print("  " + ", ".join(f"L1_{b}" for b in FEATURE_BASES))
    print("  + division, size_class")
    print("\nOUTCOMES (NaN = not a candidate or no valid full future window):")
    print("  " + ", ".join(outcome_cols))
    print("\nDO NOT USE AS FEATURES (kept only to construct outcomes/productivity):")
    print("  " + ", ".join(cfg["do_not_use_as_features"]))
    print("\nNext: python scripts/04_design_diagnostics.py")


if __name__ == "__main__":
    main()
