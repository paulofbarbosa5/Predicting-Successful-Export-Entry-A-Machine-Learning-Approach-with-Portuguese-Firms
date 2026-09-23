# 
# %%

"""06_run_policy_baselines.py
Evaluate simple targeting rules and transparent logit baselines before machine learning.

Outputs:
    outputs/predictions/policy_baseline_predictions_<outcome>.parquet
    outputs/tables/policy_baseline_targeting_<outcome>.csv
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from _common import load_config, ensure_dirs, resolve
from _model_common import (
    compute_metrics,
    get_features,
    load_model_sample,
    make_preprocessor,
    predict_scores,
    slug,
)

RANKING_RULES = {
    "random": None,
    "productivity_ranking": "labor_productivity_vab",
    "employment_ranking": "log_employment",
    "assets_ranking": "log_assets",
    "import_experience_ranking": "import_experience",
    "eu_import_share_ranking": "imports_comu",
}

LOGIT_RULES = {
    "logit_productivity_size_sector": "simple_policy_t",
    "logit_productivity_size_sector_imports": "simple_policy_imports_t",
    "logit_productivity_size_sector_imports_financial": "policy_financial_t",
}


def percentile_score(x: pd.Series) -> np.ndarray:
    # Higher values imply higher target priority. Missing values get the median rank.
    s = pd.to_numeric(x, errors="coerce")
    med = s.median()
    s = s.fillna(med)
    return s.rank(method="average", pct=True).to_numpy(dtype="float64")


def run_logit(train_valid: pd.DataFrame, test: pd.DataFrame, outcome: str, feature_set: str, seed: int) -> np.ndarray:
    features = get_features(feature_set)
    pipe = Pipeline([
        ("prep", make_preprocessor(features, scale_numeric=True)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
    ])
    pipe.fit(train_valid[features], train_valid[outcome].astype(int))
    return predict_scores(pipe, test[features])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default="entry_from_zero_to_success_t1")
    ap.add_argument("--split-designs", default="time", help="Comma-separated split designs without 'split_' prefix. Default paper output reports the main chronological split only.")
    ap.add_argument("--seed", type=int, default=123)
    args, _ = ap.parse_known_args()

    cfg = load_config()
    ensure_dirs(cfg)
    fid, yr = cfg["columns"]["firm_id"], cfg["columns"]["year"]
    tables = resolve(cfg["paths"]["tables_dir"])
    preds_dir = resolve(cfg["paths"]["predictions_dir"])
    df = load_model_sample(cfg, args.outcome)

    split_designs = [s.strip() for s in args.split_designs.split(",") if s.strip()]
    pred_rows = []
    metric_rows = []

    for split in split_designs:
        split_col = f"split_{split}"
        if split_col not in df.columns:
            print(f"[baselines] skipping {split}: {split_col} not found. Run script 05 with this split.")
            continue
        train_valid = df[df[split_col].isin(["train", "valid"])].copy()
        test = df[df[split_col] == "test"].copy()
        if train_valid.empty or test.empty:
            print(f"[baselines] skipping {split}: train/valid or test is empty.")
            continue
        y_test = test[args.outcome].astype(int).to_numpy()

        # Ranking rules.
        for rule, col in RANKING_RULES.items():
            if rule == "random":
                rng = np.random.default_rng(args.seed)
                score = rng.random(len(test))
            elif col not in test.columns:
                print(f"[baselines] skipping {rule}: column {col} not found")
                continue
            else:
                score = percentile_score(test[col])

            pred = test[[fid, yr, "size_class", "division", args.outcome]].copy()
            pred = pred.rename(columns={args.outcome: "y_true"})
            pred["score"] = score
            pred["method"] = rule
            pred["split_design"] = split
            pred["feature_set"] = "ranking"
            pred_rows.append(pred)

            metrics = compute_metrics(y_test, score)
            metrics.update({"method": rule, "split_design": split, "feature_set": "ranking"})
            metric_rows.append(metrics)

        # Logit policy rules.
        for rule, fs in LOGIT_RULES.items():
            try:
                score = run_logit(train_valid, test, args.outcome, fs, args.seed)
            except Exception as exc:
                print(f"[baselines] skipping {rule}: {exc}")
                continue
            pred = test[[fid, yr, "size_class", "division", args.outcome]].copy()
            pred = pred.rename(columns={args.outcome: "y_true"})
            pred["score"] = score
            pred["method"] = rule
            pred["split_design"] = split
            pred["feature_set"] = fs
            pred_rows.append(pred)

            metrics = compute_metrics(y_test, score)
            metrics.update({"method": rule, "split_design": split, "feature_set": fs})
            metric_rows.append(metrics)

    if not pred_rows:
        sys.exit("No baseline predictions were generated.")

    pred_df = pd.concat(pred_rows, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)

    pred_path = preds_dir / f"policy_baseline_predictions_{slug(args.outcome)}.parquet"
    tab_path = tables / f"policy_baseline_targeting_{slug(args.outcome)}.csv"
    pred_df.to_parquet(pred_path, index=False)
    metrics_df.to_csv(tab_path, index=False)

    print("\n=== POLICY BASELINE METRICS ===")
    cols = ["split_design", "method", "n", "positives", "pr_auc", "roc_auc", "precision_at_10", "recall_at_10", "lift_at_10"]
    print(metrics_df[[c for c in cols if c in metrics_df.columns]].to_string(index=False))
    print(f"\nSaved {pred_path}")
    print(f"Saved {tab_path}")


if __name__ == "__main__":
    main()
