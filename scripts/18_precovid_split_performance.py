"""18_precovid_split_performance.py
Pre-COVID split performance for the headline outcome.

Referee point (COVID confound): the main test outcomes fall in 2020-2021 and the
post-2018 split is also pandemic-era. 05_make_splits.py already defines a pre-COVID
split (default_time_windows(..., design="pre_covid")), and split_diagnostics reports
its event counts, but its PREDICTIVE performance is not reported. This script computes
it, reusing the project pipeline so the numbers are consistent with the headline tables.

It reconstructs the baseline candidate sample (entry_from_zero_to_success_t1) and the
pre-COVID time split exactly as 05_make_splits.py does, tunes and fits each reported
model on train+valid via _model_common.tune_on_validation, predicts on test, and reports
test-sample metrics from _model_common.compute_metrics with 95% firm-clustered bootstrap
intervals on the lifts (same cluster bootstrap as 08_evaluate_and_bootstrap.py):

        ROC-AUC, PR-AUC, lift@5, lift@10

It also prints the pre-COVID train/valid/test event counts and test base rate so you can
confirm they match split_diagnostics (the pre-COVID test entry rate).

Run from the scripts folder (after 03_build_entry_panel.py has produced the panel):
        python 18_precovid_split_performance.py
        python 18_precovid_split_performance.py --bootstrap 2000

Output:
        <tables_dir>/tab_precovid_performance_v1.tex
        <tables_dir>/precovid_performance.csv
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
    default_time_windows,
    horizon_for_outcome,
)

OUTCOME = "entry_from_zero_to_success_t1"
FEATURE_SET_NAME = "main_t"
MODELS_TO_REPORT = ["logit", "random_forest", "gradient_boosting"]

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


def ci(boot, key):
    if boot.empty or key not in boot.columns or boot[key].notna().sum() <= 10:
        return (np.nan, np.nan)
    return float(boot[key].quantile(0.025)), float(boot[key].quantile(0.975))


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

    h = horizon_for_outcome(OUTCOME)
    max_valid_year = int(panel[yr].max()) - h
    sample = panel[panel[OUTCOME].notna() & (panel[yr] <= max_valid_year)].copy()
    sample[OUTCOME] = sample[OUTCOME].astype(int)

    min_y, max_y = int(sample[yr].min()), int(panel[yr].max())
    try:
        windows = default_time_windows(min_y, max_y, h, design="pre_covid")
    except Exception as exc:
        sys.exit(f"Could not build pre-COVID windows: {exc}")
    print(f"[pre-covid] feature set: {_FEATURES_SOURCE}")
    print(f"[pre-covid] candidate-year windows: {windows}")

    sample["_split"] = assign_time(sample, yr, windows)
    train = sample[sample["_split"] == "train"]
    valid = sample[sample["_split"] == "valid"]
    test = sample[sample["_split"] == "test"]

    print("\nPre-COVID split diagnostics (confirm against split_diagnostics):")
    for nm, d in (("train", train), ("valid", valid), ("test", test)):
        print(f"  {nm:5} events = {int(d[OUTCOME].sum()):>6}   (n = {len(d):,})")
    base_rate = 100.0 * float(test[OUTCOME].mean()) if len(test) else float("nan")
    print(f"  test base rate = {base_rate:.1f}%\n")

    if len(test) == 0 or train[OUTCOME].nunique() < 2 or valid[OUTCOME].nunique() < 2:
        sys.exit("Pre-COVID split has an empty test set or single-class train/valid.")

    rows = []
    for model_name in MODELS_TO_REPORT:
        try:
            model, _, _ = tune_on_validation(
                model_name, train, valid, FEATURES, OUTCOME, seed=args.seed, quick=args.quick
            )
        except ValueError as exc:
            print(f"  [skip] {model_name}: {exc}")
            continue
        scored = test.copy()
        scored["score"] = predict_scores(model, test[FEATURES])
        point = compute_metrics(scored[OUTCOME].to_numpy(), scored["score"].to_numpy())
        boot = firm_cluster_bootstrap(scored, fid, "score", OUTCOME, args.bootstrap, args.seed)
        l5lo, l5hi = ci(boot, "lift_at_5")
        l10lo, l10hi = ci(boot, "lift_at_10")
        rows.append({
            "model": model_name,
            "roc_auc": float(point["roc_auc"]),
            "pr_auc": float(point["pr_auc"]),
            "lift5": float(point.get("lift_at_5", np.nan)), "lift5_lo": l5lo, "lift5_hi": l5hi,
            "lift10": float(point.get("lift_at_10", np.nan)), "lift10_lo": l10lo, "lift10_hi": l10hi,
        })
        print(f"  [{model_name:>16}] ROC={point['roc_auc']:.3f}  "
              f"lift@5={point.get('lift_at_5', float('nan')):.2f} [{l5lo:.2f}, {l5hi:.2f}]  "
              f"lift@10={point.get('lift_at_10', float('nan')):.2f} [{l10lo:.2f}, {l10hi:.2f}]")

    if not rows:
        sys.exit("No models could be evaluated on the pre-COVID split.")

    df = pd.DataFrame(rows)
    csv_path = tables / "precovid_performance.csv"
    df.to_csv(csv_path, index=False)

    def fmt_ci(p, lo, hi):
        if np.isnan(p):
            return "--"
        if np.isnan(lo) or np.isnan(hi):
            return f"{p:.2f}"
        return f"{p:.2f} [{lo:.2f}, {hi:.2f}]"

    lines = [
        r"\begin{table}[t]",
        r"\captionof{table}{Pre-COVID split: forward-looking prediction performance}"
        r" \label{tab:precovid_performance}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & ROC-AUC & PR-AUC & Lift@5 & Lift@10 \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['model'].replace('_', ' ')} & {r['roc_auc']:.3f} & {r['pr_auc']:.3f} & "
            f"{fmt_ci(r['lift5'], r['lift5_lo'], r['lift5_hi'])} & "
            f"{fmt_ci(r['lift10'], r['lift10_lo'], r['lift10_hi'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\note{Note: Pre-COVID split, in which all test outcomes ($t+1$) fall before the "
        r"2020--2021 pandemic period; the split is the one diagnosed in "
        r"Table~\ref{tab:split_diagnostics}. The candidate pool and outcome are the baseline "
        r"(zero exports at $t$, successful exporter at $t+1$). The test-sample base entry rate "
        f"is {base_rate:.1f}\\%. "
        f"Bracketed quantities are 95\\% firm-clustered bootstrap intervals ({args.bootstrap} "
        r"resamples of firms). Lift is precision divided by the base entry rate in the same "
        r"test sample.}",
        r"\end{table}",
    ]
    tex_path = tables / "tab_precovid_performance_v1.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved {csv_path}")
    print(f"Saved {tex_path}")
    print(r"Input with: \input{tables/tab_precovid_performance_v1.tex}")


if __name__ == "__main__":
    main()
