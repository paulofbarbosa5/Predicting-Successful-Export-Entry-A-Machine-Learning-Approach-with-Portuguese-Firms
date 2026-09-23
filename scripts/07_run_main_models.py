"""07_run_main_models.py
Run leakage-aware forward-looking prediction models with a common validation protocol.

Default models:
    logit, elastic_net, random_forest, gradient_boosting, mlp

Outputs:
    outputs/predictions/main_model_predictions_<outcome>_<feature_set>.parquet
    outputs/tables/main_model_metrics_<outcome>_<feature_set>.csv
    outputs/tables/main_model_hyperparameters_<outcome>_<feature_set>.csv
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default="entry_from_zero_to_success_t1")
    ap.add_argument("--feature-set", default="main_t", help="main_t, main_l1, no_sales_derived_t, no_imports_t, etc.")
    ap.add_argument("--split-designs", default="time,firm_grouped_time", help="Comma-separated split designs without 'split_' prefix.")
    ap.add_argument("--models", default="logit,elastic_net,random_forest,gradient_boosting,mlp")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--quick", action="store_true", help="Use small tuning grids for debugging.")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    fid, yr = cfg["columns"]["firm_id"], cfg["columns"]["year"]
    tables = resolve(cfg["paths"]["tables_dir"])
    preds_dir = resolve(cfg["paths"]["predictions_dir"])
    df = load_model_sample(cfg, args.outcome)
    features = get_features(args.feature_set)

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        sys.exit(f"Missing feature columns in model_sample: {missing_features}")

    split_designs = [s.strip() for s in args.split_designs.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    pred_rows = []
    metric_rows = []
    hp_rows = []

    for split in split_designs:
        split_col = f"split_{split}"
        if split_col not in df.columns:
            print(f"[models] skipping {split}: {split_col} not found. Run script 05 first.")
            continue
        train = df[df[split_col] == "train"].copy()
        valid = df[df[split_col] == "valid"].copy()
        test = df[df[split_col] == "test"].copy()
        if train.empty or valid.empty or test.empty:
            print(f"[models] skipping {split}: train, valid, or test is empty.")
            continue
        if train[args.outcome].nunique() < 2 or valid[args.outcome].nunique() < 2 or test[args.outcome].nunique() < 2:
            print(f"[models] skipping {split}: at least one split has only one class.")
            continue

        for model_name in models:
            print(f"\n[models] fitting {model_name} | split={split} | feature_set={args.feature_set}")
            try:
                model, best_params, tuning = tune_on_validation(
                    model_name=model_name,
                    train=train,
                    valid=valid,
                    features=features,
                    outcome=args.outcome,
                    seed=args.seed,
                    quick=args.quick,
                )
            except Exception as exc:
                print(f"[models] FAILED {model_name} on {split}: {exc}")
                continue

            score = predict_scores(model, test[features])
            y_test = test[args.outcome].astype(int).to_numpy()
            metrics = compute_metrics(y_test, score)
            metrics.update({
                "method": model_name,
                "split_design": split,
                "feature_set": args.feature_set,
                "best_params": str(best_params),
            })
            metric_rows.append(metrics)

            hp = tuning.copy()
            hp["split_design"] = split
            hp["feature_set"] = args.feature_set
            hp["outcome"] = args.outcome
            hp["chosen"] = hp["params"].eq(str(best_params))
            hp_rows.append(hp)

            pred = test[[fid, yr, "size_class", "division", args.outcome]].copy()
            pred = pred.rename(columns={args.outcome: "y_true"})
            pred["score"] = score
            pred["method"] = model_name
            pred["split_design"] = split
            pred["feature_set"] = args.feature_set
            pred_rows.append(pred)

    if not pred_rows:
        sys.exit("No model predictions were generated. Try --quick or check split/event counts.")

    pred_df = pd.concat(pred_rows, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    hp_df = pd.concat(hp_rows, ignore_index=True) if hp_rows else pd.DataFrame()

    tag = f"{slug(args.outcome)}_{slug(args.feature_set)}"
    pred_path = preds_dir / f"main_model_predictions_{tag}.parquet"
    metrics_path = tables / f"main_model_metrics_{tag}.csv"
    hp_path = tables / f"main_model_hyperparameters_{tag}.csv"
    pred_df.to_parquet(pred_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    hp_df.to_csv(hp_path, index=False)

    print("\n=== MAIN MODEL METRICS ===")
    cols = ["split_design", "method", "n", "positives", "pr_auc", "roc_auc", "precision_at_10", "recall_at_10", "lift_at_10"]
    print(metrics_df[[c for c in cols if c in metrics_df.columns]].to_string(index=False))
    print(f"\nSaved {pred_path}")
    print(f"Saved {metrics_path}")
    print(f"Saved {hp_path}")


if __name__ == "__main__":
    main()
