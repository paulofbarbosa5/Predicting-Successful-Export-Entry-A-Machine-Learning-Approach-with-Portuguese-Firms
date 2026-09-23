"""08_evaluate_and_bootstrap.py
Fast evaluator for prediction files.

Changes relative to the first Stage-2 version:
  * default bootstrap is 0 (point estimates only) so the script finishes fast;
  * optional --bootstrap B for final confidence intervals;
  * optional --include-policy to include policy-baseline predictions;
  * optional --methods and --split-designs filters;
  * faster clustered bootstrap using numpy index resampling instead of pandas concat;
  * progress messages so the terminal does not look frozen;
  * suppresses sklearn FutureWarning from calibration-logit internals.

Outputs:
    outputs/tables/model_performance_with_ci_<outcome>.csv
    outputs/tables/calibration_deciles_<outcome>.csv
    outputs/tables/entry_rate_by_score_decile_<outcome>.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from _common import load_config, ensure_dirs, resolve
from _model_common import calibration_deciles, calibration_slope_intercept, compute_metrics, slug

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def parse_csv_arg(x: str | None) -> set[str] | None:
    if x is None or str(x).strip().lower() in {"", "all"}:
        return None
    return {s.strip() for s in str(x).split(",") if s.strip()}


def load_prediction_files(preds_dir: Path, outcome: str, feature_set: str | None, include_policy: bool) -> pd.DataFrame:
    files: list[Path] = []
    if include_policy:
        files.extend(preds_dir.glob(f"policy_baseline_predictions_{slug(outcome)}.parquet"))
    if feature_set:
        files.extend(preds_dir.glob(f"main_model_predictions_{slug(outcome)}_{slug(feature_set)}.parquet"))
    else:
        files.extend(preds_dir.glob(f"main_model_predictions_{slug(outcome)}_*.parquet"))

    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            "No prediction files found. Run scripts 06/07 first, or add --include-policy if you only have policy baselines."
        )

    frames = []
    print("[eval] loading prediction files:", flush=True)
    for p in files:
        print(f"  - {p.name}", flush=True)
        d = pd.read_parquet(p)
        d["source_file"] = p.name
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def bootstrap_group_fast(df: pd.DataFrame, fid: str, B: int, seed: int, progress_every: int = 50) -> pd.DataFrame:
    """Firm-cluster bootstrap, optimized to avoid repeated pandas concat.

    The bootstrap samples firms with replacement. When a firm is sampled more than once,
    all of its rows are repeated, as in the standard cluster bootstrap.
    """
    if B <= 0:
        return pd.DataFrame()

    work = df[[fid, "y_true", "score"]].dropna().copy()
    if work.empty or work["y_true"].nunique() < 2:
        return pd.DataFrame()

    # Factorize firm IDs and build row-index arrays by firm.
    firm_codes, firm_values = pd.factorize(work[fid], sort=False)
    n_firms = len(firm_values)
    if n_firms < 2:
        return pd.DataFrame()

    y = work["y_true"].astype(int).to_numpy()
    score = work["score"].astype(float).to_numpy()

    groups = [np.flatnonzero(firm_codes == i) for i in range(n_firms)]
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for b in range(B):
        if progress_every and (b == 0 or (b + 1) % progress_every == 0 or b + 1 == B):
            print(f"      bootstrap {b + 1}/{B}", flush=True)
        sampled = rng.integers(0, n_firms, size=n_firms)
        idx = np.concatenate([groups[i] for i in sampled])
        yb = y[idx]
        if np.unique(yb).size < 2:
            continue
        sb = score[idx]
        m = compute_metrics(yb, sb)
        m["bootstrap"] = b
        rows.append(m)

    return pd.DataFrame(rows)


def summarize_boot(point: dict, boot: pd.DataFrame, group_keys: dict) -> dict:
    out = dict(group_keys)
    for k, v in point.items():
        if not isinstance(v, (int, float, np.integer, np.floating)):
            continue
        out[k] = v
        if not boot.empty and k in boot.columns and boot[k].notna().sum() > 10:
            out[f"{k}_lo"] = float(boot[k].quantile(0.025))
            out[f"{k}_hi"] = float(boot[k].quantile(0.975))
        else:
            out[f"{k}_lo"] = np.nan
            out[f"{k}_hi"] = np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default="entry_from_zero_to_success_t1")
    ap.add_argument("--feature-set", default="main_t", help="Use 'all' to evaluate all main model feature-set files.")
    ap.add_argument("--bootstrap", type=int, default=0, help="Firm-cluster bootstrap reps. Use 0 for point estimates only.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--include-policy", action="store_true", help="Also include policy-baseline prediction files.")
    ap.add_argument("--methods", default="all", help="Comma-separated method filter, e.g. logit,gradient_boosting; default all.")
    ap.add_argument("--split-designs", default="all", help="Comma-separated split filter, e.g. time,firm_grouped_time; default all.")
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    fid = cfg["columns"]["firm_id"]
    tables = resolve(cfg["paths"]["tables_dir"])
    preds_dir = resolve(cfg["paths"]["predictions_dir"])
    feature_filter = None if args.feature_set == "all" else args.feature_set
    method_filter = parse_csv_arg(args.methods)
    split_filter = parse_csv_arg(args.split_designs)

    pred = load_prediction_files(preds_dir, args.outcome, feature_filter, include_policy=args.include_policy)
    if fid not in pred.columns:
        raise ValueError(f"Prediction files must contain firm id column {fid} for clustered bootstrap.")

    if method_filter is not None:
        pred = pred[pred["method"].isin(method_filter)].copy()
    if split_filter is not None:
        pred = pred[pred["split_design"].isin(split_filter)].copy()
    if pred.empty:
        raise ValueError("No prediction rows left after filters. Check --methods and --split-designs.")

    group_cols = ["split_design", "method", "feature_set"]
    perf_rows: list[dict] = []
    cal_rows: list[pd.DataFrame] = []
    decile_rows: list[pd.DataFrame] = []

    groups = list(pred.groupby(group_cols, dropna=False))
    print(f"[eval] evaluating {len(groups)} prediction groups; bootstrap={args.bootstrap}", flush=True)

    for i, (keys, d) in enumerate(groups, start=1):
        split_design, method, feature_set = keys
        d = d.dropna(subset=["y_true", "score"]).copy()
        if d.empty or d["y_true"].nunique() < 2:
            print(f"[eval] skipping {keys}: empty or one class only", flush=True)
            continue

        group_keys = {"split_design": split_design, "method": method, "feature_set": feature_set}
        print(
            f"\n[eval] group {i}/{len(groups)} | split={split_design} | method={method} | "
            f"feature_set={feature_set} | n={len(d):,} | positives={int(d['y_true'].sum()):,}",
            flush=True,
        )

        point = compute_metrics(d["y_true"], d["score"])
        try:
            intercept, slope = calibration_slope_intercept(d["y_true"], d["score"])
        except Exception:
            intercept, slope = np.nan, np.nan
        point["calibration_intercept"] = intercept
        point["calibration_slope"] = slope

        boot = bootstrap_group_fast(d, fid=fid, B=args.bootstrap, seed=args.seed, progress_every=args.progress_every)
        perf_rows.append(summarize_boot(point, boot, group_keys))

        cal = calibration_deciles(d)
        if not cal.empty:
            for k, v in group_keys.items():
                cal[k] = v
            cal_rows.append(cal)
            decile_rows.append(cal.rename(columns={"observed_rate": "entry_rate"}))

    perf = pd.DataFrame(perf_rows)
    cal = pd.concat(cal_rows, ignore_index=True) if cal_rows else pd.DataFrame()
    dec = pd.concat(decile_rows, ignore_index=True) if decile_rows else pd.DataFrame()

    suffix = f"{slug(args.outcome)}"
    if method_filter is not None:
        suffix += "_filtered"
    perf_path = tables / f"model_performance_with_ci_{suffix}.csv"
    cal_path = tables / f"calibration_deciles_{suffix}.csv"
    dec_path = tables / f"entry_rate_by_score_decile_{suffix}.csv"
    perf.to_csv(perf_path, index=False)
    cal.to_csv(cal_path, index=False)
    dec.to_csv(dec_path, index=False)

    print("\n=== PERFORMANCE WITH BOOTSTRAP CI ===")
    cols = [
        "split_design", "method", "feature_set", "n", "positives", "pr_auc", "pr_auc_lo", "pr_auc_hi",
        "roc_auc", "precision_at_10", "precision_at_10_lo", "precision_at_10_hi", "lift_at_10",
    ]
    print(perf[[c for c in cols if c in perf.columns]].to_string(index=False))
    print(f"\nSaved {perf_path}")
    print(f"Saved {cal_path}")
    print(f"Saved {dec_path}")


if __name__ == "__main__":
    main()
