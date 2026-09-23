"""10_interactions.py
Feature importance and fast nonlinear interaction diagnostics.

The default interaction statistic is intentionally fast: it uses model predictions on the
held-out test set, bins two features into quantiles, and measures how much of the binned
prediction surface is non-additive after subtracting the two one-way profiles. This is an
interpretability diagnostic, not a causal estimate.

Optional PDP figures can be requested with --make-pdp.

Outputs:
    outputs/tables/permutation_importance_<outcome>_<feature_set>.csv
    outputs/tables/interaction_strength_<outcome>_<feature_set>.csv
    outputs/figures/pdp_<feature1>__<feature2>_<outcome>_<feature_set>.png   (optional)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from _common import load_config, ensure_dirs, resolve
from _model_common import get_features, load_model_sample, predict_scores, slug, tune_on_validation


def choose_feature(prefix: str, feature_set: str) -> str:
    return f"L1_{prefix}" if feature_set.endswith("_l1") else prefix


def candidate_pairs(feature_set: str) -> list[tuple[str, str]]:
    prod = choose_feature("labor_productivity_vab", feature_set)
    imp = choose_feature("imports_comu", feature_set)
    imp_exp = choose_feature("import_experience", feature_set)
    emp = choose_feature("log_employment", feature_set)
    eq = choose_feature("equity_ratio", feature_set)
    cap = choose_feature("capital_intensity", feature_set)
    return [(prod, imp), (prod, imp_exp), (prod, emp), (imp, emp), (prod, eq), (prod, cap)]


def qbin(s: pd.Series, q: int) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.nunique(dropna=True) <= 2:
        return x.fillna(-999).astype(float)
    try:
        return pd.qcut(x.rank(method="first"), q=min(q, x.notna().sum()), labels=False, duplicates="drop")
    except Exception:
        return pd.Series(np.nan, index=s.index)


def fast_interaction_strength(X: pd.DataFrame, score: np.ndarray, f1: str, f2: str, bins: int = 8) -> float:
    if f1 not in X.columns or f2 not in X.columns:
        return np.nan
    d = pd.DataFrame({"f1": X[f1], "f2": X[f2], "score": score}).dropna()
    if len(d) < 50:
        return np.nan
    d["b1"] = qbin(d["f1"], bins)
    d["b2"] = qbin(d["f2"], bins)
    d = d.dropna(subset=["b1", "b2"])
    if d["b1"].nunique() < 2 or d["b2"].nunique() < 2:
        return np.nan
    grand = d["score"].mean()
    m1 = d.groupby("b1")["score"].mean() - grand
    m2 = d.groupby("b2")["score"].mean() - grand
    cell = d.groupby(["b1", "b2"])["score"].mean().reset_index()
    cell["additive"] = cell["b1"].map(m1).astype(float) + cell["b2"].map(m2).astype(float) + grand
    resid = cell["score"] - cell["additive"]
    denom = np.var(cell["score"])
    if denom <= 0:
        return np.nan
    h2 = np.var(resid) / denom
    return float(np.sqrt(max(0.0, min(1.0, h2))))


def save_pdp(model, X: pd.DataFrame, pair: tuple[str, str], out_path) -> bool:
    f1, f2 = pair
    if f1 not in X.columns or f2 not in X.columns:
        return False
    try:
        plt.figure(figsize=(7, 5))
        PartialDependenceDisplay.from_estimator(model, X, features=[pair], grid_resolution=15)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return True
    except Exception as exc:
        plt.close()
        print(f"[interactions] PDP failed for {pair}: {exc}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default="entry_from_zero_to_success_t1")
    ap.add_argument("--feature-set", default="main_t")
    ap.add_argument("--split-design", default="time")
    ap.add_argument("--model", default="gradient_boosting")
    ap.add_argument("--max-test-rows", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--make-pdp", action="store_true", help="Create PDP figures; slower on large test sets.")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    tables = resolve(cfg["paths"]["tables_dir"])
    figs = resolve(cfg["paths"]["figures_dir"])
    df = load_model_sample(cfg, args.outcome)
    features = get_features(args.feature_set)
    split_col = f"split_{args.split_design}"
    if split_col not in df.columns:
        sys.exit(f"{split_col} not found. Run script 05 for this split design.")

    train = df[df[split_col] == "train"].copy()
    valid = df[df[split_col] == "valid"].copy()
    test = df[df[split_col] == "test"].copy()
    if train.empty or valid.empty or test.empty:
        sys.exit("Train, valid, or test is empty.")
    if test[args.outcome].nunique() < 2:
        sys.exit("Test set has only one class; cannot compute interactions reliably.")

    model, best_params, _ = tune_on_validation(args.model, train, valid, features, args.outcome, seed=args.seed, quick=args.quick)
    if len(test) > args.max_test_rows:
        test_eval = test.sample(n=args.max_test_rows, random_state=args.seed)
    else:
        test_eval = test.copy()
    X_eval = test_eval[features]
    y_eval = test_eval[args.outcome].astype(int)
    score_eval = predict_scores(model, X_eval)

    # Permutation importance by PR-AUC.
    try:
        pi = permutation_importance(
            model,
            X_eval,
            y_eval,
            scoring="average_precision",
            n_repeats=3 if args.quick else 8,
            random_state=args.seed,
            n_jobs=1,
        )
        imp = pd.DataFrame({
            "feature": features,
            "importance_mean": pi.importances_mean,
            "importance_std": pi.importances_std,
            "model": args.model,
            "split_design": args.split_design,
            "feature_set": args.feature_set,
            "outcome": args.outcome,
            "best_params": str(best_params),
        }).sort_values("importance_mean", ascending=False)
    except Exception as exc:
        print(f"[interactions] permutation importance failed: {exc}")
        imp = pd.DataFrame()
    imp_path = tables / f"permutation_importance_{slug(args.outcome)}_{slug(args.feature_set)}.csv"
    imp.to_csv(imp_path, index=False)

    pair_rows = []
    for f1, f2 in candidate_pairs(args.feature_set):
        if f1 not in features or f2 not in features:
            continue
        h = fast_interaction_strength(X_eval, score_eval, f1, f2, bins=5 if args.quick else 8)
        pair_rows.append({
            "feature_1": f1,
            "feature_2": f2,
            "interaction_strength_fast": h,
            "model": args.model,
            "split_design": args.split_design,
            "feature_set": args.feature_set,
            "outcome": args.outcome,
        })
        if args.make_pdp:
            fig_path = figs / f"pdp_{slug(f1)}__{slug(f2)}_{slug(args.outcome)}_{slug(args.feature_set)}.png"
            save_pdp(model, X_eval.copy(), (f1, f2), fig_path)

    inter = pd.DataFrame(pair_rows).sort_values("interaction_strength_fast", ascending=False)
    inter_path = tables / f"interaction_strength_{slug(args.outcome)}_{slug(args.feature_set)}.csv"
    inter.to_csv(inter_path, index=False)

    print("\n=== PERMUTATION IMPORTANCE ===")
    print(imp.head(20).to_string(index=False) if not imp.empty else "No importance table produced.")
    print("\n=== FAST INTERACTION STRENGTH ===")
    print(inter.to_string(index=False) if not inter.empty else "No interaction table produced.")
    print(f"\nSaved {imp_path}")
    print(f"Saved {inter_path}")
    if args.make_pdp:
        print(f"PDP figures saved in {figs}")
    else:
        print("PDP figures skipped. Re-run with --make-pdp for figures.")


if __name__ == "__main__":
    main()
