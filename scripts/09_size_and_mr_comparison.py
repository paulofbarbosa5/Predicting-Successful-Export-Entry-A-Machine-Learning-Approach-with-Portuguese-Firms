"""09_size_and_mr_comparison.py
Use the full Portuguese firm universe to compare export-readiness prediction by size.

This script produces two related outputs:
  1) Evaluation of saved predictions by size/sample group.
  2) Optional retraining of one selected model within sample groups.

Outputs:
    outputs/tables/size_group_prediction_performance_<outcome>.csv
    outputs/tables/size_group_retrained_performance_<outcome>.csv
    outputs/figures/score_distribution_by_size_<outcome>.png
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def load_predictions(preds_dir, outcome: str, feature_set: str) -> pd.DataFrame:
    files = []
    files.extend(preds_dir.glob(f"main_model_predictions_{slug(outcome)}_{slug(feature_set)}.parquet"))
    files.extend(preds_dir.glob(f"policy_baseline_predictions_{slug(outcome)}.parquet"))
    if not files:
        raise FileNotFoundError("No predictions found. Run scripts 06 and 07 first.")
    return pd.concat([pd.read_parquet(p).assign(source_file=p.name) for p in files], ignore_index=True)


def evaluate_saved_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, method, feature_set), d0 in pred.groupby(["split_design", "method", "feature_set"], dropna=False):
        for sample, sizes in SAMPLE_GROUPS.items():
            d = d0[d0["size_class"].isin(sizes)].dropna(subset=["y_true", "score"])
            if d.empty:
                continue
            metrics = compute_metrics(d["y_true"], d["score"])
            metrics.update({
                "sample": sample,
                "split_design": split,
                "method": method,
                "feature_set": feature_set,
                "size_classes": "+".join(sizes),
                "median_score": float(d["score"].median()),
                "p90_score": float(d["score"].quantile(0.90)),
            })
            rows.append(metrics)
    return pd.DataFrame(rows)


def retrain_by_group(df: pd.DataFrame, outcome: str, feature_set: str, split_design: str, model_name: str, seed: int, quick: bool, min_test_events: int) -> pd.DataFrame:
    features = get_features(feature_set)
    split_col = f"split_{split_design}"
    rows = []
    for sample, sizes in SAMPLE_GROUPS.items():
        d = df[df["size_class"].isin(sizes)].copy()
        train = d[d[split_col] == "train"]
        valid = d[d[split_col] == "valid"]
        test = d[d[split_col] == "test"]
        if train.empty or valid.empty or test.empty:
            rows.append({"sample": sample, "method": model_name, "status": "empty split"})
            continue
        test_events = int(test[outcome].sum())
        if test_events < min_test_events:
            rows.append({"sample": sample, "method": model_name, "status": f"too few test events ({test_events})", "test_events": test_events})
            continue
        if train[outcome].nunique() < 2 or valid[outcome].nunique() < 2 or test[outcome].nunique() < 2:
            rows.append({"sample": sample, "method": model_name, "status": "one class in split", "test_events": test_events})
            continue
        try:
            model, best_params, _ = tune_on_validation(model_name, train, valid, features, outcome, seed=seed, quick=quick)
            score = predict_scores(model, test[features])
            metrics = compute_metrics(test[outcome].astype(int), score)
            metrics.update({
                "sample": sample,
                "size_classes": "+".join(sizes),
                "method": model_name,
                "feature_set": feature_set,
                "split_design": split_design,
                "status": "estimated",
                "best_params": str(best_params),
            })
            rows.append(metrics)
        except Exception as exc:
            rows.append({"sample": sample, "method": model_name, "status": f"failed: {exc}", "test_events": test_events})
    return pd.DataFrame(rows)


def score_distribution_figure(pred: pd.DataFrame, outcome: str, out_path) -> None:
    d = pred[(pred["split_design"] == "time") & (pred["method"] == "gradient_boosting")].copy()
    if d.empty:
        d = pred.copy()
    d = d[d["size_class"].isin(["micro", "small", "medium", "large"])]
    if d.empty:
        return
    plt.figure(figsize=(8, 5))
    # Use default matplotlib colors; no explicit color choices.
    labels = []
    arrays = []
    for s in ["micro", "small", "medium", "large"]:
        vals = d.loc[d["size_class"] == s, "score"].dropna().to_numpy()
        if len(vals):
            labels.append(s)
            arrays.append(vals)
    if arrays:
        plt.boxplot(arrays, labels=labels, showfliers=False)
        plt.ylabel("Predicted export-entry score")
        plt.xlabel("Firm size class")
        plt.title("Distribution of export-readiness scores by size class")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default="entry_from_zero_to_success_t1")
    ap.add_argument("--feature-set", default="main_t")
    ap.add_argument("--retrain-model", default="gradient_boosting")
    ap.add_argument("--split-design", default="time")
    ap.add_argument("--min-test-events", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    tables = resolve(cfg["paths"]["tables_dir"])
    preds_dir = resolve(cfg["paths"]["predictions_dir"])
    figs = resolve(cfg["paths"]["figures_dir"])

    pred = load_predictions(preds_dir, args.outcome, args.feature_set)
    by_group = evaluate_saved_predictions(pred)
    by_group_path = tables / f"size_group_prediction_performance_{slug(args.outcome)}.csv"
    by_group.to_csv(by_group_path, index=False)

    fig_path = figs / f"score_distribution_by_size_{slug(args.outcome)}.png"
    score_distribution_figure(pred, args.outcome, fig_path)

    df = load_model_sample(cfg, args.outcome)
    retrained = retrain_by_group(df, args.outcome, args.feature_set, args.split_design, args.retrain_model, args.seed, args.quick, args.min_test_events)
    retrained_path = tables / f"size_group_retrained_performance_{slug(args.outcome)}.csv"
    retrained.to_csv(retrained_path, index=False)

    print("\n=== SIZE-GROUP PERFORMANCE FROM SAVED PREDICTIONS ===")
    cols = ["sample", "split_design", "method", "n", "positives", "pr_auc", "precision_at_10", "recall_at_10", "lift_at_10"]
    print(by_group[[c for c in cols if c in by_group.columns]].head(40).to_string(index=False))
    print("\n=== RETRAINED SIZE-GROUP PERFORMANCE ===")
    print(retrained.to_string(index=False))
    print(f"\nSaved {by_group_path}")
    print(f"Saved {retrained_path}")
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
