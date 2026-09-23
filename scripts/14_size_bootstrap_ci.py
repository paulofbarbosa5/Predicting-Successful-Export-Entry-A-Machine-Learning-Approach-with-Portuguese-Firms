"""13_size_bootstrap_ci.py

Firm-clustered bootstrap confidence intervals for size-stratified prediction results.

This addresses the referee concern that the small-firm result is based on a finite
number of entry events. The bootstrap is conditional on the fitted model: it resamples
held-out test firms with replacement and recomputes ranking metrics on the resampled
predictions.

Default: gradient_boosting, main_t, time split.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from _common import load_config, ensure_dirs, resolve
from _model_common import (
    compute_metrics,
    get_features,
    load_model_sample,
    predict_scores,
    slug,
    tune_on_validation,
)

SAMPLE_GROUPS = {
    "all_firms": ["micro", "small", "medium", "large"],
    "micro_small": ["micro", "small"],
    "medium_large": ["medium", "large"],
    "micro": ["micro"],
    "small": ["small"],
    "medium": ["medium"],
    "large": ["large"],
}

METRIC_COLS = [
    "roc_auc", "pr_auc", "brier", "log_loss",
    "precision_at_5", "recall_at_5", "lift_at_5",
    "precision_at_10", "recall_at_10", "lift_at_10",
]


def parse_csv_arg(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def clean_feature_list(features: list[str], drop: Iterable[str]) -> list[str]:
    drop = set(drop)
    return [f for f in features if f not in drop]


def bootstrap_prediction_metrics(test_pred: pd.DataFrame, outcome_col: str, score_col: str, firm_col: str, B: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firms = test_pred[firm_col].dropna().unique()
    rows = []
    grouped = {k: v for k, v in test_pred.groupby(firm_col, sort=False)}
    for b in range(B):
        sampled = rng.choice(firms, size=len(firms), replace=True)
        boot = pd.concat([grouped[f] for f in sampled], ignore_index=True)
        if boot[outcome_col].nunique() < 2:
            continue
        m = compute_metrics(boot[outcome_col].astype(int), boot[score_col].astype(float))
        rows.append({k: m.get(k, np.nan) for k in METRIC_COLS})
    return pd.DataFrame(rows)


def summarize_ci(point: dict, boot: pd.DataFrame, sample: str, model_name: str, split_design: str, feature_set: str, status: str, best_params: str | None, test_events: int) -> dict:
    row = dict(point)
    row.update({
        "sample": sample,
        "method": model_name,
        "split_design": split_design,
        "feature_set": feature_set,
        "status": status,
        "best_params": best_params,
        "test_events": test_events,
    })
    for c in METRIC_COLS:
        if c in boot.columns and len(boot):
            row[f"{c}_lo"] = float(boot[c].quantile(0.025))
            row[f"{c}_hi"] = float(boot[c].quantile(0.975))
        else:
            row[f"{c}_lo"] = np.nan
            row[f"{c}_hi"] = np.nan
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default="entry_from_zero_to_success_t1")
    ap.add_argument("--feature-set", default="main_t")
    ap.add_argument("--split-design", default="time")
    ap.add_argument("--model", default="gradient_boosting")
    ap.add_argument("--samples", default="all_firms,micro_small,medium_large,micro,small,medium,large")
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--min-test-events", type=int, default=20)
    ap.add_argument("--drop-features", default="book_leverage")
    ap.add_argument("--size-var", default="size_class",
                    help="Column used to STRATIFY firms into size groups (e.g. size_class or size_class_ec). "
                         "The model features are unchanged: the employment-based size_class always stays in the feature set.")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    fid, yr = cfg["columns"]["firm_id"], cfg["columns"]["year"]
    tables = resolve(cfg["paths"]["tables_dir"])
    preds_dir = resolve(cfg["paths"]["predictions_dir"])
    df = load_model_sample(cfg, args.outcome)
    split_col = f"split_{args.split_design}"
    if split_col not in df.columns:
        raise SystemExit(f"{split_col} not found. Run script 05 first.")

    if args.size_var not in df.columns:
        raise SystemExit(f"size variable '{args.size_var}' not found in the model sample. "
                         f"Rebuild the panel (script 03) so it is present.")
    features = clean_feature_list(get_features(args.feature_set), parse_csv_arg(args.drop_features))
    samples = [s for s in parse_csv_arg(args.samples) if s in SAMPLE_GROUPS]
    rows = []
    pred_rows = []
    for sample in samples:
        sizes = SAMPLE_GROUPS[sample]
        d = df[df[args.size_var].isin(sizes)].copy()
        train = d[d[split_col] == "train"].copy()
        valid = d[d[split_col] == "valid"].copy()
        test = d[d[split_col] == "test"].copy()
        test_events = int(test[args.outcome].sum()) if len(test) else 0
        print(f"[size-ci] {sample}: test n={len(test):,}, events={test_events}")
        if train.empty or valid.empty or test.empty or test_events < args.min_test_events:
            # Still record the group's size/event counts so the table can show the row
            # (e.g. the Large cell: 19 candidates, 2 entrants, 10.5% baseline) even though
            # the group is too sparse to estimate a model and bootstrap CIs.
            n_test = int(len(test))
            rows.append({"sample": sample, "n": n_test, "positives": test_events,
                         "prevalence": (test_events / n_test) if n_test else float("nan"),
                         "status": f"too few test events ({test_events})", "test_events": test_events})
            continue
        if train[args.outcome].nunique() < 2 or valid[args.outcome].nunique() < 2 or test[args.outcome].nunique() < 2:
            rows.append({"sample": sample, "status": "one class in split", "test_events": test_events})
            continue
        model, best_params, _ = tune_on_validation(args.model, train, valid, features, args.outcome, seed=args.seed, quick=args.quick)
        score = predict_scores(model, test[features])
        point = compute_metrics(test[args.outcome].astype(int), score)
        keep_id_cols = [fid, yr, "size_class", "division", args.outcome]
        if args.size_var not in keep_id_cols:
            keep_id_cols.insert(3, args.size_var)
        tp = test[keep_id_cols].copy()
        tp = tp.rename(columns={args.outcome: "y_true"})
        tp["score"] = score
        tp["sample"] = sample
        tp["method"] = args.model
        tp["split_design"] = args.split_design
        tp["feature_set"] = args.feature_set
        pred_rows.append(tp)
        boot = bootstrap_prediction_metrics(tp, "y_true", "score", fid, B=args.bootstrap, seed=args.seed)
        rows.append(summarize_ci(point, boot, sample, args.model, args.split_design, args.feature_set, "estimated", str(best_params), test_events))

    res = pd.DataFrame(rows)
    size_tag = "" if args.size_var == "size_class" else f"_{slug(args.size_var)}"
    tag = f"{slug(args.outcome)}_{slug(args.feature_set)}_{slug(args.split_design)}_{slug(args.model)}{size_tag}"
    out = tables / f"size_group_bootstrap_ci_{tag}.csv"
    res.to_csv(out, index=False)
    if pred_rows:
        pred = pd.concat(pred_rows, ignore_index=True)
        pred.to_parquet(preds_dir / f"size_group_bootstrap_predictions_{tag}.parquet", index=False)
    print("\n=== SIZE-GROUP BOOTSTRAP CI ===")
    cols = ["sample", "n", "positives", "roc_auc", "roc_auc_lo", "roc_auc_hi", "pr_auc", "pr_auc_lo", "pr_auc_hi", "lift_at_10", "lift_at_10_lo", "lift_at_10_hi", "precision_at_10", "precision_at_10_lo", "precision_at_10_hi", "status"]
    print(res[[c for c in cols if c in res.columns]].to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
