"""20_shap_horizon_comparison.py

Two-panel SHAP comparison: one-year outcome (t+1) versus two-year outcome
(t+1:t+2), both with winsorized productivity.

Purpose (revision point): the current Figure 4 shows 14 features, several of
which are near-zero or near-duplicates, and the paper now reports the two-year
outcome in parallel with the headline. This script produces the replacement
figure: two side-by-side panels showing the top-N mean |SHAP| features for the
Gradient Boosting model under each outcome window, on a shared x-axis, so the
predictive-signal ranking can be compared across horizons directly.

The SHAP plumbing (winsorization caps fit on the training window, TreeExplainer
on the transformed design matrix, one-hot groups summed within observation
before taking mean absolute values) is copied VERBATIM from
13_run_winsorized_shap.py so the one-year panel reproduces the published
Figure 4 / Table 6 ranking. Sanity anchor for the one-year panel:
division ~ 0.212, log_assets ~ 0.112, equity_ratio ~ 0.093 (Table 6).

Prerequisites: model_sample parquets for BOTH outcomes. The first exists from
the main pipeline; the second is one command:

    python scripts/05_make_splits.py --outcome entry_from_zero_to_success_t1_t2

Run from the package root:

    python scripts\\20_shap_horizon_comparison.py
    python scripts\\20_shap_horizon_comparison.py --quick        (smoke test)
    python scripts\\20_shap_horizon_comparison.py --top-n 10     (more bars)

Output:
    outputs/tables/shap_horizon_comparison_<tag>.csv     (all features, both horizons)
    outputs/tables/shap_horizon_metrics_<tag>.csv        (fit diagnostics per horizon)
    outputs/figures/fig_shap_horizon_comparison_<tag>.png
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
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

OUTCOMES = [
    ("entry_from_zero_to_success_t1", "One-year window (t+1)"),
    ("entry_from_zero_to_success_t1_t2", "Two-year window (t+1:t+2)"),
]

DISPLAY_NAMES = {
    "division": "Sector (NACE division)",
    "log_assets": "Assets (log)",
    "equity_ratio": "Equity ratio",
    "labor_productivity_vab": "Labour productivity (VAB)",
    "capital_intensity": "Capital intensity",
    "profit_margin": "Profit margin",
    "wage_per_worker": "Wage per worker",
    "imports_comu": "EU import intensity",
    "imports_extra": "Extra-EU import intensity",
    "import_experience": "Import experience",
    "roa": "Return on assets",
    "log_employment": "Employment (log)",
    "size_class": "Size class",
    "negative_equity": "Negative equity",
}


# ----------------------------------------------------------------------------
# Functions below are copied verbatim from 13_run_winsorized_shap.py so the
# aggregation and winsorization are identical to the published Figure 4.
# If you change them there, change them here too.
# ----------------------------------------------------------------------------
def parse_csv_arg(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def clean_feature_list(features: list[str], drop: Iterable[str]) -> list[str]:
    drop = set(drop)
    out = [f for f in features if f not in drop]
    removed = [f for f in features if f in drop]
    if removed:
        print(f"[shap-compare] dropped collinear/redundant features: {removed}")
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
            rows.append({"feature": f, "lower_quantile": lower, "upper_quantile": upper,
                         "lower_cap": lo, "upper_cap": hi})
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
        vals = vals[-1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, -1]
    return vals


def grouped_shap_importance(shap_vals: np.ndarray, transformed_names: list[str],
                            original_features: list[str]) -> pd.DataFrame:
    rows = []
    base_map = [base_feature_from_transformed(n, original_features) for n in transformed_names]
    for f in sorted(set(base_map)):
        idx = [i for i, b in enumerate(base_map) if b == f]
        if not idx:
            continue
        contrib = shap_vals[:, idx].sum(axis=1)
        rows.append({"feature": f, "mean_abs_shap": float(np.mean(np.abs(contrib)))})
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
# ----------------------------------------------------------------------------
# End of verbatim block.
# ----------------------------------------------------------------------------


def run_one(cfg, outcome: str, args) -> tuple[pd.DataFrame, dict]:
    """Winsorize, fit, and compute grouped SHAP importance for one outcome."""
    split_col = f"split_{args.split_design}"
    try:
        df = load_model_sample(cfg, outcome)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\nBuild it first with:\n"
            f"    python scripts/05_make_splits.py --outcome {outcome}"
        )
    if split_col not in df.columns:
        raise SystemExit(
            f"{split_col} not found in model_sample for {outcome}. "
            f"Re-run: python scripts/05_make_splits.py --outcome {outcome}"
        )

    features = clean_feature_list(get_features(args.feature_set),
                                  parse_csv_arg(args.drop_features))
    clip_features = [f for f in parse_csv_arg(args.clip_features) if f in features]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise SystemExit(f"Missing features in model sample for {outcome}: {missing}")

    train_idx = df[split_col].eq("train")
    valid_idx = df[split_col].eq("valid")
    test_idx = df[split_col].eq("test")
    if not (train_idx.any() and valid_idx.any() and test_idx.any()):
        raise SystemExit(f"Empty train/valid/test for {split_col} in {outcome}.")

    caps = fit_winsor_caps(df[train_idx], clip_features, args.lower, args.upper)
    print(f"\n[shap-compare] {outcome}: winsor caps (train quantiles "
          f"{args.lower:.2f}/{args.upper:.2f}):")
    print(caps.to_string(index=False))

    df_w = df.copy()
    if not caps.empty:
        df_w.loc[:, clip_features] = apply_winsor_caps(df_w[clip_features], caps)[clip_features]
    train = df_w[train_idx].copy()
    valid = df_w[valid_idx].copy()
    test = df_w[test_idx].copy()

    print(f"[shap-compare] fitting {args.model} | outcome={outcome} | "
          f"split={args.split_design} | n_test={len(test):,}")
    model, best_params, _ = tune_on_validation(
        model_name=args.model, train=train, valid=valid,
        features=features, outcome=outcome, seed=args.seed, quick=args.quick,
    )

    score = predict_scores(model, test[features])
    y = test[outcome].astype(int).to_numpy()
    m = compute_metrics(y, score)
    print(f"[shap-compare] {outcome}: ROC={m['roc_auc']:.3f}  PR={m['pr_auc']:.3f}  "
          f"lift@10={m.get('lift_at_10', float('nan')):.2f}  "
          f"(winsorized fit diagnostics)")

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
    transformed_names = onehot_feature_names(
        prep, [f for f in features if f in {"division", "size_class"}]
    )
    if len(transformed_names) != X_trans.shape[1]:
        transformed_names = [f"x{i}" for i in range(X_trans.shape[1])]

    explainer = shap.TreeExplainer(est)
    shap_vals = get_shap_array(explainer, X_trans)
    imp = grouped_shap_importance(shap_vals, transformed_names, features)

    metrics_row = {
        "outcome": outcome, "model": args.model, "split_design": args.split_design,
        "n_test": int(m["n"]), "positives": int(m["positives"]),
        "roc_auc": float(m["roc_auc"]), "pr_auc": float(m["pr_auc"]),
        "lift_at_10": float(m.get("lift_at_10", np.nan)),
        "shap_rows": int(len(shap_idx)), "best_params": str(best_params),
    }
    return imp, metrics_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gradient_boosting",
                    choices=["gradient_boosting", "random_forest"])
    ap.add_argument("--feature-set", default="main_t")
    ap.add_argument("--split-design", default="time")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--lower", type=float, default=0.01)
    ap.add_argument("--upper", type=float, default=0.99)
    ap.add_argument("--clip-features", default="labor_productivity_vab")
    ap.add_argument("--drop-features", default="book_leverage")
    ap.add_argument("--max-shap-rows", type=int, default=10000)
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    tables = resolve(cfg["paths"]["tables_dir"])
    figures = resolve(cfg["paths"]["figures_dir"])
    figures.mkdir(parents=True, exist_ok=True)

    tag = f"{slug(args.feature_set)}_{slug(args.split_design)}_{slug(args.model)}_winsor"

    results, metric_rows = [], []
    for outcome, label in OUTCOMES:
        imp, mrow = run_one(cfg, outcome, args)
        results.append((outcome, label, imp))
        metric_rows.append(mrow)

    # ---- combined CSV: all features, both horizons, with ranks ----
    merged = None
    for outcome, label, imp in results:
        suffix = "_t1" if outcome.endswith("_t1") else "_t1t2"
        d = imp.rename(columns={"mean_abs_shap": f"mean_abs_shap{suffix}"}).copy()
        d[f"rank{suffix}"] = np.arange(1, len(d) + 1)
        merged = d if merged is None else merged.merge(d, on="feature", how="outer")
    csv_path = tables / f"shap_horizon_comparison_{tag}.csv"
    merged.to_csv(csv_path, index=False)

    metrics_path = tables / f"shap_horizon_metrics_{tag}.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)

    # ---- two-panel figure, shared x-axis ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    xmax = 0.0
    for ax, (outcome, label, imp) in zip(axes, results):
        d = imp.head(args.top_n).iloc[::-1]
        labels = [DISPLAY_NAMES.get(f, f) for f in d["feature"]]
        ax.barh(labels, d["mean_abs_shap"])
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("mean |SHAP value|")
        xmax = max(xmax, float(d["mean_abs_shap"].max()))
    for ax in axes:
        ax.set_xlim(0, xmax * 1.08)
    fig.tight_layout()
    fig_path = figures / f"fig_shap_horizon_comparison_{tag}.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print("\n[shap-compare] top features per horizon:")
    for outcome, label, imp in results:
        top = ", ".join(f"{r.feature} {r.mean_abs_shap:.3f}"
                        for r in imp.head(5).itertuples())
        print(f"  {label}: {top}")
    print("\n[shap-compare] sanity anchor (one-year panel vs Table 6): "
          "division ~0.212, log_assets ~0.112, equity_ratio ~0.093")
    print(f"\nSaved {csv_path}")
    print(f"Saved {metrics_path}")
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
