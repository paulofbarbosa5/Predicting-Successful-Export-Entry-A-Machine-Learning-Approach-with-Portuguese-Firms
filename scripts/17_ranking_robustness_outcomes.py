"""17_ranking_robustness_outcomes.py
Ranking-metric robustness across alternative outcome definitions.

Referee point: Table 2 (tab_entry_outcomes) shows how the candidate count and base
rate move across outcome definitions, but never whether the model still RANKS well
under them. This script fills that gap, reusing the project pipeline so the numbers
are consistent with the headline tables.

For each alternative outcome -- which already exist as columns in entry_panel.parquet
built by 03_build_entry_panel.py -- it reconstructs the modeling sample exactly as
05_make_splits.py does (candidates with a non-missing outcome and a valid future
window, main chronological time split), tunes and fits the headline model on
train+valid via _model_common.tune_on_validation, predicts on test, and reports
test-sample ranking metrics from _model_common.compute_metrics:

        ROC-AUC, PR-AUC, lift@5, lift@10  (+ candidate FY, entries, base rate)

ROC-AUC is the headline comparison metric because lift = precision / base rate
compresses mechanically as the base rate rises, so a falling lift across broader
pools is expected and is not evidence of a worse ranking.

Run from the scripts folder (after 03_build_entry_panel.py has produced the panel):
        python 17_ranking_robustness_outcomes.py
        python 17_ranking_robustness_outcomes.py --include-horizons
        python 17_ranking_robustness_outcomes.py --model gradient_boosting

Output:
        <tables_dir>/tab_outcome_robustness_v1.tex
        <tables_dir>/outcome_robustness.csv
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
    slug,
    default_time_windows,
    horizon_for_outcome,
)

# --- headline model and feature set used in the paper ----------------------
MODEL_NAME = "random_forest"
FEATURE_SET_NAME = "main_t"

# Resolve the feature list. Prefer the project's own registry if it exists, so
# this script uses exactly the same predictors as 07_run_main_models.py.
try:
    from _model_common import FEATURE_SETS  # type: ignore
    FEATURES = list(FEATURE_SETS[FEATURE_SET_NAME])
    _FEATURES_SOURCE = "FEATURE_SETS['%s']" % FEATURE_SET_NAME
except Exception:
    # Fallback: contemporaneous features as built in 03_build_entry_panel.py.
    FEATURE_BASES = [
        "labor_productivity_vab", "imports_comu", "imports_extra", "import_experience",
        "log_employment", "log_assets", "capital_intensity", "wage_per_worker",
        "equity_ratio", "roa", "profit_margin", "negative_equity",
    ]
    FEATURES = FEATURE_BASES + ["division", "size_class"]
    _FEATURES_SOURCE = "fallback (contemporaneous FEATURE_BASES + division, size_class)"

# Outcome columns that share the one-year horizon (identical test window, so
# ROC-AUC is fully comparable across these rows).
PRIMARY_OUTCOMES = [
    ("entry_from_zero_to_success_t1",
     r"Zero exports $\to$ successful exporter ($t{+}1$, baseline)"),
    ("entry_from_zero_to_positive_t1",
     r"Zero exports $\to$ any positive exports ($t{+}1$)"),
    ("entry_from_near_zero_to_success_t1",
     r"Near-zero exports $\to$ successful exporter ($t{+}1$)"),
    ("transition_below10_to_success_t1",
     r"Below 10\% intensity $\to$ successful exporter ($t{+}1$)"),
    ("transition_positive_below10_to_success_t1",
     r"Positive but $<$10\% $\to$ successful exporter ($t{+}1$)"),
]

# Multi-year horizons. These use horizon-specific time windows (see
# default_time_windows), so their test window differs from the one-year rows.
EXTENDED_OUTCOMES = [
    ("entry_from_zero_to_success_t1_t2",
     r"Zero exports $\to$ successful within $t{+}1$:$t{+}2$"),
    ("entry_from_zero_to_success_t1_t3",
     r"Zero exports $\to$ successful within $t{+}1$:$t{+}3$"),
    ("entry_from_near_zero_to_success_t1_t2",
     r"Near-zero exports $\to$ successful within $t{+}1$:$t{+}2$"),
    ("transition_below10_to_success_t1_t2",
     r"Below 10\% intensity $\to$ successful within $t{+}1$:$t{+}2$"),
]


def assign_time(df: pd.DataFrame, yr: str, windows: dict) -> pd.Series:
    """Same time-split assignment used in 05_make_splits.py."""
    out = pd.Series(pd.NA, index=df.index, dtype="string")
    for split, years in windows.items():
        out.loc[df[yr].isin(years)] = split
    return out


def evaluate_outcome(panel, outcome, label, fid, yr, model_name, seed, quick):
    if outcome not in panel.columns:
        print(f"  [skip] '{outcome}' not in panel.")
        return None

    h = horizon_for_outcome(outcome)
    max_valid_year = int(panel[yr].max()) - h
    sample = panel[panel[outcome].notna() & (panel[yr] <= max_valid_year)].copy()
    if sample[outcome].nunique() < 2:
        print(f"  [skip] '{outcome}': single outcome class.")
        return None
    sample[outcome] = sample[outcome].astype(int)

    min_y, max_y = int(sample[yr].min()), int(panel[yr].max())
    windows = default_time_windows(min_y, max_y, h, design="main")
    sample["_split"] = assign_time(sample, yr, windows)
    train = sample[sample["_split"] == "train"]
    valid = sample[sample["_split"] == "valid"]
    test = sample[sample["_split"] == "test"]

    if len(test) == 0 or train[outcome].nunique() < 2 or valid[outcome].nunique() < 2:
        print(f"  [skip] '{outcome}': empty test or single-class train/valid.")
        return None

    try:
        model, best_params, _ = tune_on_validation(
            model_name, train, valid, FEATURES, outcome, seed=seed, quick=quick
        )
    except ValueError as exc:
        print(f"  [skip] '{outcome}': {exc}")
        return None

    scores = predict_scores(model, test[FEATURES])
    m = compute_metrics(test[outcome].to_numpy(), scores)
    row = {
        "outcome": outcome,
        "label": label,
        "horizon": h,
        "cand_fy_test": int(m["n"]),
        "entries_test": int(m["positives"]),
        "base_rate_pct": 100.0 * float(m["prevalence"]),
        "roc_auc": float(m["roc_auc"]),
        "pr_auc": float(m["pr_auc"]),
        "lift5": float(m.get("lift_at_5", np.nan)),
        "lift10": float(m.get("lift_at_10", np.nan)),
    }
    print(f"  [{model_name}] {label[:46]:46}  N={row['cand_fy_test']:>6}  "
          f"base={row['base_rate_pct']:4.1f}%  ROC={row['roc_auc']:.3f}  "
          f"lift@10={row['lift10']:.2f}")
    return row


def to_latex(df: pd.DataFrame, model_name: str, include_horizons: bool) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\captionof{table}{Ranking robustness across outcome definitions}"
        r" \label{tab:outcome_robustness}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Outcome definition & Cand. FY & Entries & Base (\%)"
        r" & ROC-AUC & PR-AUC & Lift@5 & Lift@10 \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['label']} & {int(r['cand_fy_test']):,} & {int(r['entries_test']):,} & "
            f"{r['base_rate_pct']:.1f} & {r['roc_auc']:.3f} & {r['pr_auc']:.3f} & "
            f"{r['lift5']:.2f} & {r['lift10']:.2f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    horizon_note = (
        r" Rows with a multi-year window ($t{+}1$:$t{+}k$) use horizon-specific time "
        r"windows and are therefore not evaluated on the one-year test window."
        if include_horizons else ""
    )
    lines.append(
        r"\note{Note: Each row reconstructs the candidate sample and main chronological "
        r"split as in the headline design and reports test-sample ranking metrics for the "
        + model_name.replace("_", " ") + r" model. Cand.\ FY and Entries are test-sample "
        r"candidate firm-years and successful entrants; Base is the test-sample base entry "
        r"rate. ROC-AUC is the preferred comparison metric because lift mechanically "
        r"compresses as the base rate rises (lift $=$ precision$/$base rate), whereas "
        r"ROC-AUC is invariant to the base rate." + horizon_note +
        r" Full-sample candidate counts are in Table~\ref{tab:entry_outcomes_v2}.}"
    )
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--include-horizons", action="store_true",
                    help="Also evaluate the multi-year (t+1:t+2, t+1:t+3) outcomes.")
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

    print(f"[robustness] model={args.model}  feature set: {_FEATURES_SOURCE}")
    print(f"[robustness] features ({len(FEATURES)}): {FEATURES}\n")

    outcomes = list(PRIMARY_OUTCOMES)
    if args.include_horizons:
        outcomes += EXTENDED_OUTCOMES

    rows = []
    for outcome, label in outcomes:
        r = evaluate_outcome(panel, outcome, label, fid, yr, args.model, args.seed, args.quick)
        if r is not None:
            rows.append(r)

    if not rows:
        sys.exit("No outcomes could be evaluated. Check the panel and feature set.")

    df = pd.DataFrame(rows)
    csv_path = tables / "outcome_robustness.csv"
    df.to_csv(csv_path, index=False)
    tex_path = tables / "tab_outcome_robustness_v1.tex"
    tex_path.write_text(to_latex(df, args.model, args.include_horizons), encoding="utf-8")

    print(f"\nSaved {csv_path}")
    print(f"Saved {tex_path}")
    print(r"Input with: \input{tables/tab_outcome_robustness_v1.tex}")


if __name__ == "__main__":
    main()
