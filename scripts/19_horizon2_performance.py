"""19_horizon2_performance.py
Two-year-window export-entry performance: zero exports at t -> successful
exporter within t+1:t+2.

Purpose: present the two-year outcome IN PARALLEL with the headline one-year
outcome (Table 3). The candidate pool is identical (zero exports at t); only the
outcome window changes. Candidate firm-years must have observed outcomes in BOTH
t+1 and t+2 (window_event_full in 03_build_entry_panel.py), so the candidate
sample is smaller than the one-year sample and conditions on two-year survival
and reporting. Test outcomes necessarily extend into 2020-2021.

The script reconstructs the modeling sample and the main chronological split for
horizon 2 exactly as 05_make_splits.py would (default_time_windows with h=2),
tunes and fits Logit, Random Forest and Gradient Boosting with the common
protocol (tune_on_validation), and reports test-sample metrics with 95%
firm-clustered bootstrap intervals.

Sanity anchors (from Table 2 of the paper):
    total candidate firm-years = 52,549 ; total entries = 4,310 (8.2%)
    test entries = 1,278

Run from the package root (entry_panel.parquet must exist, i.e. after script 03):
        python scripts\\19_horizon2_performance.py
        python scripts\\19_horizon2_performance.py --bootstrap 2000   (final CIs)
        python scripts\\19_horizon2_performance.py --quick            (smoke test)

Output:
        outputs/tables/horizon2_performance.csv
        outputs/tables/horizon2_decile_gradient.csv
        outputs/tables/tab_horizon2_panel_v1.tex   (standalone table)
        + console rows formatted for pasting as a fourth panel of Table 3
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from _common import load_config, ensure_dirs, resolve
from _model_common import (
    tune_on_validation,
    predict_scores,
    compute_metrics,
    calibration_deciles,
    default_time_windows,
    horizon_for_outcome,
)

OUTCOME = "entry_from_zero_to_success_t1_t2"
FEATURE_SET_NAME = "main_t"
MODELS_TO_REPORT = ["logit", "random_forest", "gradient_boosting"]
DISPLAY = {"logit": "Logit", "random_forest": "Random Forest",
           "gradient_boosting": "Gradient Boosting"}

try:
    from _model_common import FEATURE_SETS  # type: ignore
    FEATURES = list(FEATURE_SETS[FEATURE_SET_NAME])
    _FEATURES_SOURCE = "FEATURE_SETS['%s']" % FEATURE_SET_NAME
except Exception:
    FEATURE_BASES = [
        "labor_productivity_vab", "imports_comu", "imports_extra", "import_experience",
        "log_employment", "log_assets", "capital_intensity", "wage_per_worker",
        "equity_ratio", "roa", "profit_margin", "negative_equity",
    ]
    FEATURES = FEATURE_BASES + ["division", "size_class"]
    _FEATURES_SOURCE = "fallback (contemporaneous FEATURE_BASES + division, size_class)"


def assign_time(df: pd.DataFrame, yr: str, windows: dict) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="string")
    for split, years in windows.items():
        out.loc[df[yr].isin(years)] = split
    return out


def firm_cluster_bootstrap(test, fid, score_col, y_col, B, seed):
    """Firm-cluster bootstrap of compute_metrics; mirrors 08_evaluate_and_bootstrap."""
    work = test[[fid, y_col, score_col]].dropna().copy()
    if work.empty or work[y_col].nunique() < 2:
        return pd.DataFrame()
    codes, vals = pd.factorize(work[fid], sort=False)
    n_firms = len(vals)
    if n_firms < 2:
        return pd.DataFrame()
    groups = [np.flatnonzero(codes == i) for i in range(n_firms)]
    y = work[y_col].astype(int).to_numpy()
    s = work[score_col].astype(float).to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(B):
        sampled = rng.integers(0, n_firms, size=n_firms)
        idx = np.concatenate([groups[i] for i in sampled])
        yb = y[idx]
        if np.unique(yb).size < 2:
            continue
        rows.append(compute_metrics(yb, s[idx]))
    return pd.DataFrame(rows)


def ci(boot: pd.DataFrame, key: str):
    if boot.empty or key not in boot.columns or boot[key].notna().sum() <= 10:
        return (np.nan, np.nan)
    return float(boot[key].quantile(0.025)), float(boot[key].quantile(0.975))


def fmt_ci(p, lo, hi, nd=2):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    if np.isnan(lo) or np.isnan(hi):
        return f"{p:.{nd}f}"
    return f"{p:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    fid, yr = cfg["columns"]["firm_id"], cfg["columns"]["year"]
    processed = resolve(cfg["paths"]["processed_dir"])
    tables = resolve(cfg["paths"]["tables_dir"])

    panel_path = processed / "entry_panel.parquet"
    if not panel_path.exists():
        sys.exit(f"{panel_path} not found. Run scripts 01-03 first.")
    panel = pd.read_parquet(panel_path)
    if OUTCOME not in panel.columns:
        sys.exit(f"{OUTCOME} not found in entry_panel.parquet. Re-run script 03.")

    h = horizon_for_outcome(OUTCOME)
    print(f"[horizon2] outcome={OUTCOME}  horizon={h}")
    print(f"[horizon2] feature set: {_FEATURES_SOURCE}")

    max_valid_year = int(panel[yr].max()) - h
    sample = panel[panel[OUTCOME].notna() & (panel[yr] <= max_valid_year)].copy()
    sample[OUTCOME] = sample[OUTCOME].astype(int)

    n_all, e_all = len(sample), int(sample[OUTCOME].sum())
    print(f"[horizon2] candidate firm-years = {n_all:,}  entries = {e_all:,} "
          f"({100*e_all/n_all:.1f}%)   <- expect 52,549 / 4,310 (8.2%) from Table 2")

    min_y, max_y = int(sample[yr].min()), int(panel[yr].max())
    windows = default_time_windows(min_y, max_y, h, design="main")
    print(f"[horizon2] candidate-year windows: {windows}")

    sample["_split"] = assign_time(sample, yr, windows)
    train = sample[sample["_split"] == "train"]
    valid = sample[sample["_split"] == "valid"]
    test = sample[sample["_split"] == "test"]

    print("\nSplit diagnostics:")
    for nm, d in (("train", train), ("valid", valid), ("test", test)):
        print(f"  {nm:5} events = {int(d[OUTCOME].sum()):>6}   (n = {len(d):,})")
    base_rate = 100.0 * float(test[OUTCOME].mean()) if len(test) else float("nan")
    print(f"  test base rate = {base_rate:.1f}%   <- expect 1,278 test events from Table 2\n")

    if len(test) == 0 or train[OUTCOME].nunique() < 2 or valid[OUTCOME].nunique() < 2:
        sys.exit("Horizon-2 split has an empty test set or single-class train/valid.")

    test_years = sorted(int(v) for v in test[yr].unique())
    o_min, o_max = min(test_years) + 1, max(test_years) + h

    rows, dec_frames = [], []
    for model_name in MODELS_TO_REPORT:
        try:
            model, best, _ = tune_on_validation(
                model_name, train, valid, FEATURES, OUTCOME, seed=args.seed, quick=args.quick
            )
        except ValueError as exc:
            print(f"  [skip] {model_name}: {exc}")
            continue
        scored = test.copy()
        scored["score"] = predict_scores(model, test[FEATURES])
        point = compute_metrics(scored[OUTCOME].to_numpy(), scored["score"].to_numpy())
        boot = firm_cluster_bootstrap(scored, fid, "score", OUTCOME,
                                      50 if args.quick else args.bootstrap, args.seed)
        l5lo, l5hi = ci(boot, "lift_at_5")
        l10lo, l10hi = ci(boot, "lift_at_10")
        rows.append({
            "model": model_name,
            "pr_auc": float(point["pr_auc"]),
            "roc_auc": float(point["roc_auc"]),
            "p5": 100.0 * float(point.get("precision_at_5", np.nan)),
            "lift5": float(point.get("lift_at_5", np.nan)),
            "lift5_lo": l5lo, "lift5_hi": l5hi,
            "p10": 100.0 * float(point.get("precision_at_10", np.nan)),
            "lift10": float(point.get("lift_at_10", np.nan)),
            "lift10_lo": l10lo, "lift10_hi": l10hi,
        })
        print(f"  [{model_name:>17}] ROC={point['roc_auc']:.3f}  PR={point['pr_auc']:.3f}  "
              f"lift@5={fmt_ci(rows[-1]['lift5'], l5lo, l5hi)}  "
              f"lift@10={fmt_ci(rows[-1]['lift10'], l10lo, l10hi)}")

        dec = calibration_deciles(scored.rename(columns={OUTCOME: "y_true"}))
        if not dec.empty:
            dec["model"] = model_name
            dec_frames.append(dec)

    if not rows:
        sys.exit("No models could be evaluated.")
    df = pd.DataFrame(rows)

    csv_path = tables / "horizon2_performance.csv"
    df.to_csv(csv_path, index=False)
    if dec_frames:
        dec_path = tables / "horizon2_decile_gradient.csv"
        pd.concat(dec_frames, ignore_index=True).to_csv(dec_path, index=False)
        print(f"\nSaved {dec_path}")

    # ---- standalone LaTeX table (points + CIs on the lifts) ----
    test_years_str = ", ".join(str(t) for t in test_years)
    lines = [
        r"\begin{table}[!ht]",
        r"\caption{Two-year entry window: forward-looking prediction performance}",
        r"\label{tab:horizon2_performance}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & PR-AUC & ROC-AUC & P@5 (\%) & Lift@5 & P@10 (\%) & Lift@10 \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{DISPLAY[r['model']]} & {r['pr_auc']:.3f} & {r['roc_auc']:.3f} & "
            f"{r['p5']:.1f} & {fmt_ci(r['lift5'], r['lift5_lo'], r['lift5_hi'])} & "
            f"{r['p10']:.1f} & {fmt_ci(r['lift10'], r['lift10_lo'], r['lift10_hi'])} \\\\"
        )
    n_boot = 50 if args.quick else args.bootstrap
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\note{Note: The outcome is successful export entry within $t{+}1$:$t{+}2$ among "
        r"firms with zero exports in year $t$. Candidate firm-years require observed "
        r"outcomes in both $t{+}1$ and $t{+}2$, so the sample conditions on two-year "
        r"survival and reporting. The split uses the main chronological design for a "
        f"two-year horizon: candidate test years are {test_years_str}, so test outcomes "
        f"are observed in {o_min}--{o_max}. The test-sample baseline entry rate is "
        f"{base_rate:.1f}\\%. "
        r"P@5 and P@10 denote the observed entry rate among the top 5\% and 10\% of "
        r"scored firms; lift is precision divided by the baseline rate. Bracketed "
        f"quantities are 95\\% firm-clustered bootstrap confidence intervals ({n_boot} "
        f"resamples of firms, random seed {args.seed}).}}",
        r"\end{table}",
    ]
    tex_path = tables / "tab_horizon2_panel_v1.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- paste-ready rows for a fourth panel of Table 3 (point estimates) ----
    print("\nRows to paste as a fourth panel of Table 3 "
          "(header: Time split, two-year entry window (baseline "
          f"{base_rate:.1f}%)):\n")
    print(r"\multicolumn{7}{l}{\textit{Time split, two-year entry window "
          f"(baseline entry rate {base_rate:.1f}\\%)}}}} \\\\")
    for _, r in df.iterrows():
        print(f"{DISPLAY[r['model']]} & {r['pr_auc']:.3f} & {r['roc_auc']:.3f} & "
              f"{r['p5']:.1f} & {r['lift5']:.2f} & {r['p10']:.1f} & {r['lift10']:.2f} \\\\")

    print(f"\nSaved {csv_path}")
    print(f"Saved {tex_path}")
    print(r"Standalone table: \input{tables/tab_horizon2_panel_v1.tex}")


if __name__ == "__main__":
    main()
