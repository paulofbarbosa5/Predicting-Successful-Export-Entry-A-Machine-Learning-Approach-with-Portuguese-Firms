"""15_rf_shap.py

Compute Random Forest SHAP diagnostics for the main time split.

Outputs:
    outputs/models/rf_<outcome>_<feature_set>_<split>.joblib
    outputs/tables/rf_shap_importance_<...>.csv
    outputs/paper/tables/tab_rf_shap_importance_v3.tex
    outputs/figures/fig_rf_shap_importance_<...>.png
    outputs/figures/fig_rf_shap_dependence_productivity_capital_<...>.png
    overleaf/tables/tab_rf_shap_importance_v3.tex
    overleaf/figures/fig_rf_shap_importance.png
    overleaf/figures/fig_rf_shap_dependence_productivity_capital.png
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_config, ensure_dirs, resolve  # noqa: E402
from _model_common import (  # noqa: E402
    compute_metrics,
    get_features,
    load_model_sample,
    predict_scores,
    slug,
    tune_on_validation,
)


def ensure_extra_dirs(cfg: dict) -> tuple[Path, Path, Path, Path, Path]:
    tables = resolve(cfg["paths"]["tables_dir"])
    figs = resolve(cfg["paths"]["figures_dir"])
    preds = resolve(cfg["paths"]["predictions_dir"])
    models = resolve(cfg["paths"].get("models_dir", "outputs/models"))
    paper_tables = Path("outputs/paper/tables")
    overleaf_tables = Path("overleaf/tables")
    overleaf_figs = Path("overleaf/figures")
    for p in [tables, figs, preds, models, paper_tables, overleaf_tables, overleaf_figs]:
        p.mkdir(parents=True, exist_ok=True)
    return tables, figs, preds, models, paper_tables


def transformed_feature_names(pipe, features: list[str]) -> list[str]:
    """Return feature names after the fitted preprocessor.

    The project preprocessor contains FunctionTransformers, so sklearn's
    get_feature_names_out may fail in some environments. This function manually
    builds names from the numeric features and one-hot categorical features.
    """
    prep = pipe.named_steps["prep"]
    names: list[str] = []
    for name, trans, cols in prep.transformers_:
        if name == "remainder" or trans == "drop":
            continue
        cols = list(cols)
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            try:
                enc = trans.named_steps["onehot"]
                cat_names = enc.get_feature_names_out(cols).tolist()
            except Exception:
                cat_names = []
                enc = trans.named_steps.get("onehot")
                if enc is not None and hasattr(enc, "categories_"):
                    for c, cats in zip(cols, enc.categories_):
                        cat_names.extend([f"{c}_{v}" for v in cats])
                else:
                    cat_names.extend(cols)
            names.extend(cat_names)
    return names


def get_positive_class_shap(explainer, X_trans):
    """Return SHAP values for the positive class, robust to shap versions."""
    sv = explainer.shap_values(X_trans)
    if isinstance(sv, list):
        return sv[1] if len(sv) > 1 else sv[0]
    arr = np.asarray(sv)
    if arr.ndim == 3:
        # Common shape: n x features x classes
        if arr.shape[-1] == 2:
            return arr[:, :, 1]
        # Alternative shape: classes x n x features
        if arr.shape[0] == 2:
            return arr[1, :, :]
    return arr


def fit_random_forest_for_split(df: pd.DataFrame, outcome: str, feature_set: str, split: str, seed: int, quick: bool):
    features = get_features(feature_set)
    split_col = f"split_{split}"
    if split_col not in df.columns:
        raise ValueError(f"{split_col} not found. Run scripts/05_make_splits.py first.")
    train = df[df[split_col] == "train"].copy()
    valid = df[df[split_col] == "valid"].copy()
    test = df[df[split_col] == "test"].copy()
    if train.empty or valid.empty or test.empty:
        raise ValueError(f"Empty train/valid/test split for {split}.")
    model, best_params, tuning = tune_on_validation(
        model_name="random_forest",
        train=train,
        valid=valid,
        features=features,
        outcome=outcome,
        seed=seed,
        quick=quick,
    )
    return model, best_params, tuning, test, features


def run_rf_shap(cfg: dict, args) -> None:
    try:
        import shap
    except ImportError as exc:
        raise ImportError("The `shap` package is required. Run: python -m pip install shap") from exc

    tables, figs, _, models, paper_tables = ensure_extra_dirs(cfg)
    df = load_model_sample(cfg, args.outcome)
    model, best_params, tuning, test, features = fit_random_forest_for_split(
        df, args.outcome, args.feature_set, args.split_design, args.seed, args.quick
    )
    tag = f"{slug(args.outcome)}_{slug(args.feature_set)}_{slug(args.split_design)}_random_forest"
    model_path = models / f"rf_{tag}.joblib"
    joblib.dump(model, model_path)
    print(f"[rf-shap] saved fitted model: {model_path}")

    # Evaluate and save a small metrics row.
    score = predict_scores(model, test[features])
    y = test[args.outcome].astype(int).to_numpy()
    metrics = compute_metrics(y, score)
    metrics.update({"method": "random_forest", "split_design": args.split_design, "feature_set": args.feature_set, "best_params": str(best_params)})
    pd.DataFrame([metrics]).to_csv(tables / f"rf_shap_model_metrics_{tag}.csv", index=False)

    rng = np.random.default_rng(args.seed)
    if len(test) > args.max_shap_rows:
        idx = rng.choice(test.index.to_numpy(), size=args.max_shap_rows, replace=False)
        X_shap_raw = test.loc[idx, features].copy()
    else:
        X_shap_raw = test[features].copy()

    prep = model.named_steps["prep"]
    rf = model.named_steps["model"]
    X_trans = prep.transform(X_shap_raw)
    names = transformed_feature_names(model, features)
    if X_trans.shape[1] != len(names):
        names = [f"x{j}" for j in range(X_trans.shape[1])]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(rf)
        shap_vals = get_positive_class_shap(explainer, X_trans)

    shap_vals = np.asarray(shap_vals)
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(shap_vals).mean(axis=0)})
    # Collapse one-hot division/size dummies for readability.
    collapsed = []
    for base in ["division", "size_class"]:
        m = imp["feature"].str.startswith(base + "_")
        if m.any():
            collapsed.append({"feature": base, "mean_abs_shap": imp.loc[m, "mean_abs_shap"].sum()})
            imp = imp.loc[~m].copy()
    if collapsed:
        imp = pd.concat([imp, pd.DataFrame(collapsed)], ignore_index=True)
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp_path = tables / f"rf_shap_importance_{tag}.csv"
    imp.to_csv(imp_path, index=False)
    print(f"[rf-shap] saved {imp_path}")

    # Main figure.
    top = imp.head(args.top_n).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Random Forest SHAP importance")
    plt.tight_layout()
    fig_path = figs / f"fig_rf_shap_importance_{tag}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    (Path("overleaf/figures")).mkdir(parents=True, exist_ok=True)
    shutil_path = Path("overleaf/figures/fig_rf_shap_importance.png")
    import shutil
    shutil.copy2(fig_path, shutil_path)
    print(f"[rf-shap] saved {fig_path} and {shutil_path}")

    # Dependence: productivity, coloured by capital intensity if both are present.
    def col_index(name: str) -> int | None:
        try:
            return names.index(name)
        except ValueError:
            return None

    prod_i = col_index("labor_productivity_vab")
    cap_i = col_index("capital_intensity")
    if prod_i is not None:
        x = X_trans[:, prod_i]
        y_sv = shap_vals[:, prod_i]
        plt.figure(figsize=(7, 5))
        if cap_i is not None:
            c = X_trans[:, cap_i]
            sc = plt.scatter(x, y_sv, c=c, s=7, alpha=0.45)
            cb = plt.colorbar(sc)
            cb.set_label("capital_intensity")
        else:
            plt.scatter(x, y_sv, s=7, alpha=0.45)
        plt.xlabel("labor_productivity_vab")
        plt.ylabel("SHAP value for productivity")
        plt.title("Random Forest SHAP dependence: productivity")
        plt.tight_layout()
        dep_path = figs / f"fig_rf_shap_dependence_productivity_capital_{tag}.png"
        plt.savefig(dep_path, dpi=300)
        plt.close()
        shutil.copy2(dep_path, Path("overleaf/figures/fig_rf_shap_dependence_productivity_capital.png"))
        print(f"[rf-shap] saved {dep_path}")

    # TeX fragment for appendix.
    label_map = {
        "division": "Sector (NACE division)",
        "size_class": "Size class",
        "log_assets": "Assets (log)",
        "equity_ratio": "Equity ratio",
        "labor_productivity_vab": "Labour productivity (VAB)",
        "capital_intensity": "Capital intensity",
        "profit_margin": "Profit margin",
        "imports_comu": "EU import intensity",
        "imports_extra": "Extra-EU import intensity",
        "import_experience": "Import experience",
        "roa": "Return on assets",
        "wage_per_worker": "Wage per worker",
        "log_employment": "Employment (log)",
        "negative_equity": "Negative equity",
    }
    rows = []
    for _, r in imp.head(10).iterrows():
        feat = label_map.get(r["feature"], str(r["feature"]).replace("_", " "))
        rows.append(f"{feat} & {r['mean_abs_shap']:.4f} \\")
    tex = """\\begin{table}[!ht]
\\captionof{table}{Random Forest SHAP importance} \\label{tab:rf_shap_importance}
\\centering
\\begin{tabular}{lr}
\\hline
Feature & Mean $|$SHAP$|$ \\
\\hline
""" + "\n".join(rows) + """
\\hline
\\end{tabular}
\\note{Note: Mean absolute SHAP values for the Random Forest model in the main time split. The exercise is included as a robustness check because the Random Forest attains the highest lift in the main chronological split, while the main-text SHAP interpretation uses Gradient Boosting as the representative tree-based model. SHAP values are predictive contributions to the model score and are not causal effects.}
\\end{table}
"""
    for dest in [paper_tables / "tab_rf_shap_importance_v3.tex", Path("overleaf/tables/tab_rf_shap_importance_v3.tex")]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(tex, encoding="utf-8")
        print(f"[rf-shap] saved {dest}")




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default="entry_from_zero_to_success_t1")
    ap.add_argument("--feature-set", default="main_t")
    ap.add_argument("--split-design", default="time")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-shap-rows", type=int, default=5000)
    ap.add_argument("--top-n", type=int, default=12)
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    run_rf_shap(cfg, args)


if __name__ == "__main__":
    main()
