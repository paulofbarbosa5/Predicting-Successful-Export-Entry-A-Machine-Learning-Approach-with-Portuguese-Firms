"""12_run_winsorized_shap.py

Winsorize selected predictors, re-fit the tree model, and compute SHAP diagnostics.

Why this exists:
    Earlier SHAP diagnostics showed that VAB labour productivity has extreme and
    negative values that distort the productivity-SHAP relationship. This script
    performs the robustness/interpretability exercise requested for the revision:

      1. Fit the model after clipping selected continuous predictors using only
         the training-window distribution.
      2. Evaluate the model on the held-out test window.
      3. Save the fitted model for reproducibility.
      4. Compute real SHAP values with shap.TreeExplainer.
      5. Produce paper-ready SHAP importance and dependence figures.

Default model:
    gradient_boosting, feature_set=main_t, split_design=time.

The script does not overwrite your main results. It writes separate files with
"winsorized" in the filename.
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import warnings
from pathlib import Path
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from _common import load_config, ensure_dirs, resolve
from _model_common import (
    compute_metrics,
    get_features,
    load_model_sample,
    predict_scores,
    slug,
    tune_on_validation,
)


def parse_csv_arg(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def clean_feature_list(features: list[str], drop: Iterable[str]) -> list[str]:
    drop = set(drop)
    out = [f for f in features if f not in drop]
    removed = [f for f in features if f in drop]
    if removed:
        print(f"[winsor-shap] dropped collinear/redundant features: {removed}")
    return out


def fit_winsor_caps(train: pd.DataFrame, features: list[str], lower: float, upper: float) -> pd.DataFrame:
    rows = []
    for f in features:
        if f not in train.columns:
            continue
        x = pd.to_numeric(train[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        lo = float(x.quantile(lower))
        hi = float(x.quantile(upper))
        if np.isfinite(lo) and np.isfinite(hi) and lo <= hi:
            rows.append({"feature": f, "lower_quantile": lower, "upper_quantile": upper, "lower_cap": lo, "upper_cap": hi})
    return pd.DataFrame(rows)


def apply_winsor_caps(df: pd.DataFrame, caps: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for _, r in caps.iterrows():
        f = r["feature"]
        if f in out.columns:
            x = pd.to_numeric(out[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
            out[f] = x.clip(lower=float(r["lower_cap"]), upper=float(r["upper_cap"]))
    return out


def onehot_feature_names(prep, cat_features: list[str]) -> list[str]:
    """Recover transformed feature names from the project preprocessing pipeline."""
    names: list[str] = []
    for name, transformer, cols in prep.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        cols = list(cols)
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            # pipeline: to_string -> onehot
            try:
                oh = transformer.named_steps["onehot"]
                cats = oh.categories_
                for col, cat_values in zip(cols, cats):
                    for val in cat_values:
                        names.append(f"{col}_{val}")
            except Exception:
                names.extend(cols)
        else:
            names.extend(cols)
    return names


def base_feature_from_transformed(name: str, original_features: list[str]) -> str:
    if name in original_features:
        return name
    for f in sorted(original_features, key=len, reverse=True):
        if name.startswith(f + "_"):
            return f
    return name


def get_shap_array(explainer, X_trans):
    vals = explainer.shap_values(X_trans)
    if isinstance(vals, list):
        # Binary classifiers often return [class0, class1]
        vals = vals[-1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, -1]
    return vals


def get_interaction_array(explainer, X_trans):
    vals = explainer.shap_interaction_values(X_trans)
    if isinstance(vals, list):
        vals = vals[-1]
    vals = np.asarray(vals)
    if vals.ndim == 4:
        vals = vals[:, :, :, -1]
    return vals


def grouped_shap_importance(shap_vals: np.ndarray, transformed_names: list[str], original_features: list[str]) -> pd.DataFrame:
    rows = []
    base_map = [base_feature_from_transformed(n, original_features) for n in transformed_names]
    for f in sorted(set(base_map)):
        idx = [i for i, b in enumerate(base_map) if b == f]
        if not idx:
            continue
        # Sum SHAP contributions inside the group, then take mean absolute contribution.
        contrib = shap_vals[:, idx].sum(axis=1)
        rows.append({"feature": f, "mean_abs_shap": float(np.mean(np.abs(contrib)))})
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)


def grouped_interactions(inter_vals: np.ndarray, transformed_names: list[str], original_features: list[str], top_n: int = 30) -> pd.DataFrame:
    base_map = [base_feature_from_transformed(n, original_features) for n in transformed_names]
    unique = sorted(set(base_map))
    idx_by_feature = {f: [i for i, b in enumerate(base_map) if b == f] for f in unique}
    rows = []
    for i, f1 in enumerate(unique):
        for f2 in unique[i + 1:]:
            idx1 = idx_by_feature[f1]
            idx2 = idx_by_feature[f2]
            # Sum pairwise interactions across transformed columns in both groups.
            block = inter_vals[:, idx1, :][:, :, idx2]
            contrib = block.sum(axis=(1, 2))
            rows.append({"feature_1": f1, "feature_2": f2, "mean_abs_shap_interaction": float(np.mean(np.abs(contrib)))})
    return pd.DataFrame(rows).sort_values("mean_abs_shap_interaction", ascending=False).head(top_n)


def save_bar(df: pd.DataFrame, value_col: str, label_col: str, path: Path, title: str, xlabel: str, top_n: int = 15):
    d = df.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, 5.5))
    plt.barh(d[label_col].astype(str), d[value_col])
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def dependence_plot(
    X_test_original: pd.DataFrame,
    shap_vals: np.ndarray,
    transformed_names: list[str],
    original_features: list[str],
    feature: str,
    color_feature: str,
    path: Path,
    title: str,
):
    # For numeric features, transformed name equals original name in tree models (scale_numeric=False).
    if feature not in transformed_names:
        print(f"[winsor-shap] cannot create dependence plot: transformed feature {feature} not found.")
        return
    j = transformed_names.index(feature)
    x = pd.to_numeric(X_test_original[feature], errors="coerce")
    y = shap_vals[:, j]
    c = pd.to_numeric(X_test_original[color_feature], errors="coerce") if color_feature in X_test_original.columns else None
    mask = np.isfinite(x) & np.isfinite(y)
    if c is not None:
        mask = mask & np.isfinite(c)
    plt.figure(figsize=(7.5, 5.2))
    if c is not None:
        sc = plt.scatter(x[mask], y[mask], c=c[mask], s=8, alpha=0.35)
        cb = plt.colorbar(sc)
        cb.set_label(color_feature.replace("_", " "))
    else:
        plt.scatter(x[mask], y[mask], s=8, alpha=0.35)
    plt.axhline(0, linewidth=0.8)
    plt.xlabel(feature.replace("_", " "))
    plt.ylabel(f"SHAP value for {feature.replace('_', ' ')}")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def observed_decile_profile(df_test: pd.DataFrame, feature: str, group_feature: str, outcome: str, path: Path):
    d = df_test[[feature, group_feature, outcome]].copy()
    d[feature] = pd.to_numeric(d[feature], errors="coerce")
    d = d.dropna(subset=[feature, group_feature, outcome])
    # Use duplicates='drop' because winsorization can generate ties at caps.
    d["decile"] = pd.qcut(d[feature], 10, labels=False, duplicates="drop") + 1
    prof = d.groupby(["decile", group_feature], dropna=False)[outcome].agg(["count", "mean"]).reset_index()
    prof.to_csv(path.with_suffix(".csv"), index=False)
    plt.figure(figsize=(7.5, 5.0))
    for g, dd in prof.groupby(group_feature):
        if len(dd) == 0:
            continue
        plt.plot(dd["decile"], dd["mean"] * 100, marker="o", label=f"{group_feature}={g}")
    plt.xlabel("Productivity decile (winsorized)")
    plt.ylabel("Observed successful-entry rate (%)")
    plt.title("Observed entry rate by productivity decile and import experience")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default="entry_from_zero_to_success_t1")
    ap.add_argument("--feature-set", default="main_t")
    ap.add_argument("--split-design", default="time")
    ap.add_argument("--model", default="gradient_boosting", choices=["gradient_boosting", "random_forest"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--lower", type=float, default=0.01)
    ap.add_argument("--upper", type=float, default=0.99)
    ap.add_argument("--clip-features", default="labor_productivity_vab", help="Comma-separated numeric features to winsorize using train-window quantiles.")
    ap.add_argument("--drop-features", default="book_leverage", help="Comma-separated features to remove from the feature set before fitting.")
    ap.add_argument("--max-shap-rows", type=int, default=10000)
    ap.add_argument("--max-interaction-rows", type=int, default=2500)
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    tables = resolve(cfg["paths"]["tables_dir"])
    figures = resolve(cfg["paths"]["figures_dir"])
    preds_dir = resolve(cfg["paths"]["predictions_dir"])
    models_dir = resolve(cfg["paths"].get("models_dir", "outputs/models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    fid, yr = cfg["columns"]["firm_id"], cfg["columns"]["year"]
    split_col = f"split_{args.split_design}"
    tag = f"{slug(args.outcome)}_{slug(args.feature_set)}_{slug(args.split_design)}_{slug(args.model)}_winsor"

    df = load_model_sample(cfg, args.outcome)
    if split_col not in df.columns:
        raise SystemExit(f"{split_col} not found. Run script 05 with this split design first.")

    features = get_features(args.feature_set)
    features = clean_feature_list(features, parse_csv_arg(args.drop_features))
    clip_features = [f for f in parse_csv_arg(args.clip_features) if f in features]
    if not clip_features:
        raise SystemExit("No valid clipping features found in model feature set.")
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise SystemExit(f"Missing features in model sample: {missing_features}")

    train_idx = df[split_col].eq("train")
    valid_idx = df[split_col].eq("valid")
    test_idx = df[split_col].eq("test")
    train_raw = df[train_idx].copy()
    valid_raw = df[valid_idx].copy()
    test_raw = df[test_idx].copy()
    if train_raw.empty or valid_raw.empty or test_raw.empty:
        raise SystemExit(f"Empty train/valid/test for {split_col}.")

    caps = fit_winsor_caps(train_raw, clip_features, args.lower, args.upper)
    caps_path = tables / f"winsor_caps_{tag}.csv"
    caps.to_csv(caps_path, index=False)
    print(f"[winsor-shap] saved caps: {caps_path}")
    print(caps.to_string(index=False))

    df_w = df.copy()
    df_w.loc[:, clip_features] = apply_winsor_caps(df_w[clip_features], caps)[clip_features]
    train = df_w[train_idx].copy()
    valid = df_w[valid_idx].copy()
    test = df_w[test_idx].copy()

    print(f"[winsor-shap] fitting {args.model} on winsorized data | split={args.split_design} | features={args.feature_set}")
    model, best_params, tuning = tune_on_validation(
        model_name=args.model,
        train=train,
        valid=valid,
        features=features,
        outcome=args.outcome,
        seed=args.seed,
        quick=args.quick,
    )
    print(f"[winsor-shap] best params: {best_params}")

    score = predict_scores(model, test[features])
    y = test[args.outcome].astype(int).to_numpy()
    metrics = compute_metrics(y, score)
    metrics.update({
        "method": args.model,
        "split_design": args.split_design,
        "feature_set": args.feature_set,
        "winsorized_features": ",".join(clip_features),
        "lower_quantile": args.lower,
        "upper_quantile": args.upper,
        "dropped_features": args.drop_features,
        "best_params": str(best_params),
    })
    metrics_path = tables / f"winsorized_model_metrics_{tag}.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    pred = test[[fid, yr, "size_class", "division", args.outcome] + [f for f in clip_features if f in test.columns]].copy()
    pred = pred.rename(columns={args.outcome: "y_true"})
    pred["score"] = score
    pred["method"] = args.model
    pred["split_design"] = args.split_design
    pred["feature_set"] = args.feature_set
    pred_path = preds_dir / f"winsorized_predictions_{tag}.parquet"
    pred.to_parquet(pred_path, index=False)

    model_path = models_dir / f"{tag}.joblib"
    payload = {
        "model": model,
        "features": features,
        "outcome": args.outcome,
        "split_design": args.split_design,
        "feature_set": args.feature_set,
        "winsor_caps": caps,
        "best_params": best_params,
        "metrics": metrics,
    }
    joblib.dump(payload, model_path)
    print(f"[winsor-shap] saved model: {model_path}")
    print(pd.DataFrame([metrics])[["n", "positives", "roc_auc", "pr_auc", "precision_at_10", "recall_at_10", "lift_at_10"]].to_string(index=False))

    try:
        import shap
    except ImportError as exc:
        raise SystemExit("Install SHAP first: pip install shap") from exc

    rng = np.random.default_rng(args.seed)
    shap_idx = np.arange(len(test))
    if len(shap_idx) > args.max_shap_rows:
        shap_idx = rng.choice(shap_idx, size=args.max_shap_rows, replace=False)
    X_test = test.iloc[shap_idx][features].copy()

    prep = model.named_steps["prep"]
    est = model.named_steps["model"]
    X_trans = prep.transform(X_test)
    transformed_names = onehot_feature_names(prep, [f for f in features if f in {"division", "size_class"}])
    if len(transformed_names) != X_trans.shape[1]:
        transformed_names = [f"x{i}" for i in range(X_trans.shape[1])]

    explainer = shap.TreeExplainer(est)
    shap_vals = get_shap_array(explainer, X_trans)
    shap_imp = grouped_shap_importance(shap_vals, transformed_names, features)
    shap_imp_path = tables / f"shap_importance_{tag}.csv"
    shap_imp.to_csv(shap_imp_path, index=False)
    fig_imp = figures / f"fig_shap_importance_{tag}.png"
    save_bar(shap_imp, "mean_abs_shap", "feature", fig_imp, "SHAP importance after winsorising productivity", "mean |SHAP value|", top_n=15)

    # Dependence figures focused on productivity.
    if "labor_productivity_vab" in features:
        dep1 = figures / f"fig_shap_dependence_productivity_capital_intensity_{tag}.png"
        dependence_plot(X_test, shap_vals, transformed_names, features, "labor_productivity_vab", "capital_intensity", dep1,
                        "SHAP dependence: productivity and capital intensity")
        dep2 = figures / f"fig_shap_dependence_productivity_imports_comu_{tag}.png"
        dependence_plot(X_test, shap_vals, transformed_names, features, "labor_productivity_vab", "imports_comu", dep2,
                        "SHAP dependence: productivity and EU import intensity")
        obs_fig = figures / f"fig_observed_productivity_import_experience_{tag}.png"
        observed_decile_profile(test, "labor_productivity_vab", "import_experience", args.outcome, obs_fig)

    # Interaction values are heavier. Use smaller sample.
    inter_idx = shap_idx
    if len(inter_idx) > args.max_interaction_rows:
        inter_idx = rng.choice(inter_idx, size=args.max_interaction_rows, replace=False)
    X_inter = test.iloc[inter_idx][features].copy()
    X_inter_trans = prep.transform(X_inter)
    try:
        inter_vals = get_interaction_array(explainer, X_inter_trans)
        inter_df = grouped_interactions(inter_vals, transformed_names, features, top_n=50)
        inter_path = tables / f"shap_interaction_pairs_{tag}.csv"
        inter_df.to_csv(inter_path, index=False)
        fig_inter = figures / f"fig_shap_interactions_{tag}.png"
        inter_df["pair"] = inter_df["feature_1"] + " × " + inter_df["feature_2"]
        save_bar(inter_df, "mean_abs_shap_interaction", "pair", fig_inter, "SHAP interaction ranking after winsorising productivity", "mean |SHAP interaction|", top_n=15)
    except Exception as exc:
        warnings.warn(f"SHAP interaction values failed: {exc}. Importance/dependence plots were still saved.")

    summary_path = tables / f"winsorized_shap_summary_{tag}.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Winsorized SHAP summary\n\n")
        f.write(f"Outcome: `{args.outcome}`\n\n")
        f.write(f"Feature set: `{args.feature_set}`; split: `{args.split_design}`; model: `{args.model}`\n\n")
        f.write(f"Winsorized features: `{', '.join(clip_features)}` using training quantiles {args.lower:.3f}--{args.upper:.3f}.\n\n")
        f.write("## Metrics\n\n")
        f.write(pd.DataFrame([metrics]).T.to_markdown())
        f.write("\n\n## Top SHAP features\n\n")
        f.write(shap_imp.head(15).to_markdown(index=False))
    print(f"[winsor-shap] saved SHAP importance: {shap_imp_path}")
    print(f"[winsor-shap] saved figures to: {figures}")
    print(f"[winsor-shap] saved summary: {summary_path}")


if __name__ == "__main__":
    main()
