"""Shared modeling utilities for scripts 05--10."""
from __future__ import annotations

import itertools
import json
import math
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

FEATURE_BASES = [
    "labor_productivity_vab",
    "imports_comu",
    "imports_extra",
    "import_experience",
    "log_employment",
    "log_assets",
    "capital_intensity",
    "wage_per_worker",
    "equity_ratio",
    "roa",
    "profit_margin",
    "negative_equity",
]

OUTCOME_HORIZON = {
    "entry_from_zero_to_positive_t1": 1,
    "entry_from_zero_to_success_t1": 1,
    "entry_from_zero_to_success_t1_t2": 2,
    "entry_from_zero_to_success_t1_t3": 3,
    "entry_from_near_zero_to_success_t1": 1,
    "entry_from_near_zero_to_success_t1_t2": 2,
    "transition_below10_to_success_t1": 1,
    "transition_below10_to_success_t1_t2": 2,
    "transition_positive_below10_to_success_t1": 1,
    "transition_positive_below10_to_success_t1_t2": 2,
}

FEATURE_SETS = {
    "main_t": FEATURE_BASES + ["division", "size_class"],
    "main_l1": [f"L1_{b}" for b in FEATURE_BASES] + ["division", "size_class"],
    "no_sales_derived_t": [b for b in FEATURE_BASES if b not in {"labor_productivity_vab", "profit_margin"}] + ["division", "size_class"],
    "no_sales_derived_l1": [f"L1_{b}" for b in FEATURE_BASES if b not in {"labor_productivity_vab", "profit_margin"}] + ["division", "size_class"],
    "no_imports_t": [b for b in FEATURE_BASES if b not in {"imports_comu", "imports_extra", "import_experience"}] + ["division", "size_class"],
    "no_imports_l1": [f"L1_{b}" for b in FEATURE_BASES if b not in {"imports_comu", "imports_extra", "import_experience"}] + ["division", "size_class"],
    "simple_policy_t": ["labor_productivity_vab", "log_employment", "division", "size_class"],
    "simple_policy_imports_t": ["labor_productivity_vab", "log_employment", "imports_comu", "import_experience", "division", "size_class"],
    "policy_financial_t": ["labor_productivity_vab", "log_employment", "imports_comu", "import_experience", "equity_ratio", "roa", "division", "size_class"],
}

CATEGORICAL_FEATURES = {"division", "size_class"}


def slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(s))


def horizon_for_outcome(outcome: str) -> int:
    if outcome not in OUTCOME_HORIZON:
        raise ValueError(f"Unknown outcome '{outcome}'. Known: {sorted(OUTCOME_HORIZON)}")
    return OUTCOME_HORIZON[outcome]


def default_time_windows(min_year: int, max_year: int, horizon: int, design: str = "main") -> dict[str, list[int]]:
    """Candidate-year windows; outcome years are future years implied by horizon."""
    if design == "pre_covid":
        if horizon != 1:
            # Conservative fallback: keep future within observed data and before COVID outcome years.
            test_end = min(2018, max_year - horizon)
            return {"train": list(range(min_year, test_end - 1)), "valid": [test_end - 1], "test": [test_end]}
        return {"train": list(range(min_year, 2017)), "valid": [2017], "test": [2018]}
    if design == "post2018":
        if horizon != 1:
            raise ValueError("post2018 split is only implemented for h=1 because the panel ends in 2021.")
        return {"train": [2018], "valid": [2019], "test": [2020]}

    # Main chronological split.
    if horizon == 1:
        return {"train": list(range(min_year, 2018)), "valid": [2018], "test": [2019, 2020]}
    if horizon == 2:
        return {"train": list(range(min_year, 2017)), "valid": [2017], "test": [2018, 2019]}
    if horizon == 3:
        return {"train": list(range(min_year, 2016)), "valid": [2016], "test": [2017, 2018]}
    raise ValueError("Only horizons 1, 2, 3 are supported.")


def get_features(feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{feature_set}'. Known: {sorted(FEATURE_SETS)}")
    return FEATURE_SETS[feature_set]


def split_features(features: list[str]) -> tuple[list[str], list[str]]:
    cat = [f for f in features if f in CATEGORICAL_FEATURES]
    num = [f for f in features if f not in CATEGORICAL_FEATURES]
    return num, cat


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _to_numeric_frame(X):
    """Coerce a pandas/numpy block to numeric and convert inf to NaN."""
    Xdf = pd.DataFrame(X).copy()
    for c in Xdf.columns:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
    return Xdf.replace([np.inf, -np.inf], np.nan)


def _to_string_frame(X):
    """Coerce categorical block to strings and make missingness explicit.

    This avoids a common scikit-learn failure where pandas string/category columns
    such as size_class = 'small' are accidentally sent through an imputer that tries
    to cast them to floats.
    """
    Xdf = pd.DataFrame(X).copy()
    Xdf = Xdf.astype("object").where(pd.notna(Xdf), "__missing__")
    return Xdf.astype(str)


def normalise_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize categorical and numeric columns in the modeling sample.

    - division and size_class are categorical strings;
    - all FEATURE_BASES and L1_FEATURE_BASES are numeric;
    - inf values are converted to NaN for the numeric imputer.
    """
    out = df.copy()
    for c in CATEGORICAL_FEATURES:
        if c in out.columns:
            out[c] = out[c].astype("object").where(out[c].notna(), "__missing__").astype(str)
    numeric_candidates = set(FEATURE_BASES) | {f"L1_{b}" for b in FEATURE_BASES}
    for c in numeric_candidates:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def make_preprocessor(features: list[str], scale_numeric: bool = True) -> ColumnTransformer:
    num, cat = split_features(features)
    num_steps: list[tuple[str, Any]] = [("to_numeric", FunctionTransformer(_to_numeric_frame, validate=False)),
                                      ("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    # Missing categorical values are converted to a literal '__missing__' category.
    # No SimpleImputer is used here because some sklearn/pandas combinations try to
    # coerce string columns to float and fail with values like 'small'.
    cat_pipe = Pipeline([
        ("to_string", FunctionTransformer(_to_string_frame, validate=False)),
        ("onehot", make_onehot_encoder()),
    ])
    transformers = []
    if num:
        transformers.append(("num", num_pipe, num))
    if cat:
        transformers.append(("cat", cat_pipe, cat))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

def model_space(model_name: str, seed: int = 123, quick: bool = False) -> tuple[Any, list[dict[str, Any]], bool]:
    """Return estimator, parameter grid for estimator params, and whether scaling is appropriate."""
    name = model_name.lower()
    if name == "logit":
        est = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced", random_state=seed)
        grid = [{"C": c} for c in ([1.0] if quick else [0.1, 1.0, 10.0])]
        return est, grid, True
    if name == "elastic_net":
        est = LogisticRegression(max_iter=1200, solver="saga", penalty="elasticnet", class_weight="balanced", random_state=seed)
        Cs = [1.0] if quick else [0.1, 1.0]
        l1s = [0.5] if quick else [0.2, 0.5, 0.8]
        grid = [{"C": c, "l1_ratio": l} for c in Cs for l in l1s]
        return est, grid, True
    if name == "random_forest":
        est = RandomForestClassifier(n_jobs=-1, random_state=seed, class_weight="balanced_subsample")
        grid = [{"n_estimators": 200, "max_depth": None, "min_samples_leaf": 10}] if quick else [
            {"n_estimators": 300, "max_depth": d, "min_samples_leaf": leaf}
            for d in [None, 10, 20]
            for leaf in [5, 20]
        ]
        return est, grid, False
    if name == "gradient_boosting":
        est = GradientBoostingClassifier(random_state=seed)
        grid = [{"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3}] if quick else [
            {"n_estimators": n, "learning_rate": lr, "max_depth": d}
            for n in [150, 300]
            for lr in [0.03, 0.05]
            for d in [2, 3]
        ]
        return est, grid, False
    if name == "mlp":
        est = MLPClassifier(max_iter=250, early_stopping=True, random_state=seed)
        grid = [{"hidden_layer_sizes": (32,), "alpha": 1e-4}] if quick else [
            {"hidden_layer_sizes": h, "alpha": a}
            for h in [(32,), (64, 32)]
            for a in [1e-4, 1e-3]
        ]
        return est, grid, True
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is not installed. Run `pip install xgboost` or omit xgboost.") from exc
        est = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            random_state=seed, n_jobs=-1,
        )
        grid = [{"n_estimators": 250, "max_depth": 3, "learning_rate": 0.05}] if quick else [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr}
            for n in [250, 500]
            for d in [2, 3]
            for lr in [0.03, 0.05]
        ]
        return est, grid, False
    raise ValueError(f"Unknown model '{model_name}'.")


def make_pipeline(model_name: str, features: list[str], params: dict[str, Any], seed: int = 123, quick: bool = False) -> Pipeline:
    est, _, scale = model_space(model_name, seed=seed, quick=quick)
    est = clone(est)
    est.set_params(**params)
    return Pipeline([
        ("prep", make_preprocessor(features, scale_numeric=scale)),
        ("model", est),
    ])


def tune_on_validation(
    model_name: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    outcome: str,
    seed: int = 123,
    quick: bool = False,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    """Manual validation-set tuning by PR-AUC. Returns final model fit on train+valid."""
    est, grid, _ = model_space(model_name, seed=seed, quick=quick)
    rows = []
    best_score = -np.inf
    best_params: dict[str, Any] | None = None

    Xtr, ytr = train[features], train[outcome].astype(int)
    Xva, yva = valid[features], valid[outcome].astype(int)
    if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
        raise ValueError(f"Not enough class variation for tuning {model_name}.")

    for params in grid:
        pipe = make_pipeline(model_name, features, params, seed=seed, quick=quick)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(Xtr, ytr)
        score = predict_scores(pipe, Xva)
        pr = safe_average_precision(yva, score)
        roc = safe_roc_auc(yva, score)
        rows.append({"model": model_name, "params": json.dumps(params), "valid_pr_auc": pr, "valid_roc_auc": roc})
        if np.isfinite(pr) and pr > best_score:
            best_score = pr
            best_params = params

    if best_params is None:
        best_params = grid[0]
    final_pipe = make_pipeline(model_name, features, best_params, seed=seed, quick=quick)
    train_valid = pd.concat([train, valid], ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_pipe.fit(train_valid[features], train_valid[outcome].astype(int))
    return final_pipe, best_params, pd.DataFrame(rows)


def predict_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] > 1:
            return p[:, 1]
        return np.asarray(p).ravel()
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1 / (1 + np.exp(-z))
    return np.asarray(model.predict(X)).ravel()


def safe_roc_auc(y: Iterable[int], score: Iterable[float]) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype="float64")
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return np.nan


def safe_average_precision(y: Iterable[int], score: Iterable[float]) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype="float64")
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(average_precision_score(y, score))
    except Exception:
        return np.nan


def topk_metrics(y: Iterable[int], score: Iterable[float], k: float) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype="float64")
    n = len(y)
    if n == 0:
        return {f"precision_at_{int(k*100)}": np.nan, f"recall_at_{int(k*100)}": np.nan, f"lift_at_{int(k*100)}": np.nan, f"entrants_at_{int(k*100)}": np.nan, f"targeted_at_{int(k*100)}": np.nan}
    m = max(1, int(math.ceil(k * n)))
    idx = np.argsort(-score, kind="mergesort")[:m]
    positives = int(y.sum())
    captured = int(y[idx].sum())
    precision = captured / m if m else np.nan
    recall = captured / positives if positives else np.nan
    prevalence = positives / n if n else np.nan
    lift = precision / prevalence if prevalence and prevalence > 0 else np.nan
    label = int(round(k * 100))
    return {
        f"precision_at_{label}": precision,
        f"recall_at_{label}": recall,
        f"lift_at_{label}": lift,
        f"entrants_at_{label}": captured,
        f"targeted_at_{label}": m,
    }


def compute_metrics(y: Iterable[int], score: Iterable[float], ks: tuple[float, ...] = (0.05, 0.10)) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype="float64")
    eps = 1e-15
    score = np.clip(score, eps, 1 - eps)
    out: dict[str, float] = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else np.nan,
        "roc_auc": safe_roc_auc(y, score),
        "pr_auc": safe_average_precision(y, score),
        "brier": float(brier_score_loss(y, score)) if len(np.unique(y)) > 1 else np.nan,
        "log_loss": float(log_loss(y, score, labels=[0, 1])) if len(y) else np.nan,
    }
    for k in ks:
        out.update(topk_metrics(y, score, k))
    return out


def calibration_deciles(df: pd.DataFrame, y_col: str = "y_true", score_col: str = "score", n_bins: int = 10) -> pd.DataFrame:
    d = df[[y_col, score_col]].dropna().copy()
    if d.empty:
        return pd.DataFrame()
    try:
        d["score_decile"] = pd.qcut(d[score_col].rank(method="first"), q=n_bins, labels=False) + 1
    except ValueError:
        d["score_decile"] = 1
    return (
        d.groupby("score_decile")
        .agg(n=(y_col, "size"), mean_score=(score_col, "mean"), observed_rate=(y_col, "mean"), positives=(y_col, "sum"))
        .reset_index()
    )


def calibration_slope_intercept(y: Iterable[int], score: Iterable[float]) -> tuple[float, float]:
    """Logistic calibration intercept/slope from y ~ logit(score)."""
    y = np.asarray(y).astype(int)
    score = np.clip(np.asarray(score, dtype="float64"), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    logit_score = np.log(score / (1 - score)).reshape(-1, 1)
    try:
        cal = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    except TypeError:
        cal = LogisticRegression(penalty="none", solver="lbfgs", max_iter=1000)
    try:
        cal.fit(logit_score, y)
        return float(cal.intercept_[0]), float(cal.coef_[0][0])
    except Exception:
        return np.nan, np.nan


def save_jsonlike_csv(rows: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def load_model_sample(cfg: dict, outcome: str) -> pd.DataFrame:
    from _common import resolve
    p = resolve(cfg["paths"]["processed_dir"]) / f"model_sample_{slug(outcome)}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run scripts/05_make_splits.py first.")
    return normalise_model_frame(pd.read_parquet(p))
