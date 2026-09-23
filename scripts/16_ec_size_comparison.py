"""16_ec_size_comparison.py  (v8 -- fixed-model regrouping)

Robustness of the size composition and size-stratified ranking to the size definition.

What changed in v8
------------------
The earlier version built Panel B from script 14, which RE-ESTIMATES a separate model
within each size cell. That contradicted the appendix claim that "the trained model is
held fixed across the two columns; only the grouping of firms changes." This version makes
that claim literally true:

  1. Train ONE model on the full candidate sample (employment-only `main_t` features,
     the main chronological split). This is the same model object for every size group.
  2. Predict once on the full test set -> a single export-readiness score per test firm.
  3. Regroup those SAME predictions twice -- by employment-only `size_class` and by the
     EC-style `size_class_ec` -- and compute, within each size group, the top-decile lift
     (precision@10 / group baseline) with a firm-clustered bootstrap CI.

Panel A (entrant size composition) is model-free: it just counts entrants by class under
each definition. Panel B (size-stratified lift) uses the single fixed model's scores,
regrouped. Because the model is identical across the two columns, the table isolates the
effect of the SIZE DEFINITION, holding the trained model fixed.

Note on Table 5 vs. Table A.3
-----------------------------
Table 5 (main text) re-estimates a model WITHIN each size class -- a different and
legitimate exercise. Table A.3 (this script) holds ONE full-sample model fixed and only
regroups. The two tables therefore answer different questions and their lift numbers need
not coincide; the table note states this explicitly.

Run AFTER:
  scripts/05_make_splits.py ... --include-robustness-splits
(no dependence on script 14)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from _common import load_config, ensure_dirs, resolve
from _model_common import (
    load_model_sample,
    get_features,
    tune_on_validation,
    predict_scores,
    compute_metrics,
    slug,
)
from importlib import import_module
_m14 = import_module("14_size_bootstrap_ci")
bootstrap_prediction_metrics = _m14.bootstrap_prediction_metrics
clean_feature_list = _m14.clean_feature_list
parse_csv_arg = _m14.parse_csv_arg

SIZE_ORDER = ["micro", "small", "medium", "large"]
SAMPLE_GROUPS = {
    "all_firms": ["micro", "small", "medium", "large"],
    "micro_small": ["micro", "small"],
    "micro": ["micro"],
    "small": ["small"],
    "medium": ["medium"],
}
PANEL_B_ROWS = ["all_firms", "micro_small", "micro", "small", "medium"]
PANEL_B_LABELS = {"all_firms": "All firms", "micro_small": "Micro + small",
                  "micro": "Micro", "small": "Small", "medium": "Medium"}


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def fmt_int(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{int(round(x)):,}"


def composition(sub: pd.DataFrame, col: str) -> dict:
    counts = {k: int((sub[col] == k).sum()) for k in SIZE_ORDER}
    classified = sum(counts.values())
    counts["micro_small"] = counts["micro"] + counts["small"]
    counts["classified"] = classified
    counts["unclassified"] = int(sub[col].isna().sum())
    counts["micro_small_share"] = (100.0 * counts["micro_small"] / classified) if classified else np.nan
    return counts


def group_lift(test_pred, size_col, fid, bootstrap, seed, min_events):
    out = {}
    for sample, sizes in SAMPLE_GROUPS.items():
        d = test_pred[test_pred[size_col].isin(sizes)].copy()
        ev = int(d["y_true"].sum())
        if len(d) == 0 or ev < min_events or d["y_true"].nunique() < 2:
            out[sample] = {"lift": np.nan, "lo": np.nan, "hi": np.nan,
                           "p10": np.nan, "n": len(d), "events": ev}
            continue
        point = compute_metrics(d["y_true"].astype(int), d["score"].astype(float))
        boot = bootstrap_prediction_metrics(d, "y_true", "score", fid, B=bootstrap, seed=seed)
        lo = float(boot["lift_at_10"].quantile(0.025)) if "lift_at_10" in boot and len(boot) else np.nan
        hi = float(boot["lift_at_10"].quantile(0.975)) if "lift_at_10" in boot and len(boot) else np.nan
        out[sample] = {"lift": point.get("lift_at_10", np.nan), "lo": lo, "hi": hi,
                       "p10": point.get("precision_at_10", np.nan), "n": len(d), "events": ev}
    return out


def lift_cell(rec):
    if rec is None or (isinstance(rec.get("lift"), float) and np.isnan(rec.get("lift"))):
        return "--"
    return f"{fmt(rec['lift'])} [{fmt(rec['lo'])}, {fmt(rec['hi'])}]"


def build_latex(comp_emp, comp_ec, concord, lift_emp, lift_ec, outcome, split_design, model):
    n_class = fmt_int(concord.get('n_entrants_both_classified'))
    n_same = fmt_int(concord.get('n_same_class'))
    rows_a = ""
    for k, label in [("micro", "Micro"), ("small", "Small"), ("medium", "Medium"),
                     ("large", "Large"), ("micro_small", "Micro + small")]:
        rows_a += f"{label} & {fmt_int(comp_emp.get(k))} & {fmt_int(comp_ec.get(k))} \\\\\n"
    share_emp = fmt(comp_emp.get("micro_small_share"), 1)
    share_ec = fmt(comp_ec.get("micro_small_share"), 1)
    rows_b = ""
    for k in PANEL_B_ROWS:
        rows_b += f"{PANEL_B_LABELS[k]} & {lift_cell(lift_emp.get(k))} & {lift_cell(lift_ec.get(k))} \\\\\n"
    conc = fmt(concord.get("concordance_pct"), 1)
    n_ent = fmt_int(concord.get("n_entrants"))
    return f"""% AUTO-GENERATED by scripts/16_ec_size_comparison.py (v8, fixed-model regrouping).
% Outcome={outcome}; split={split_design}; model={model}. Do not edit by hand.
\\begin{{table}}[!htbp]
\\captionof{{table}}{{Robustness of the entrant size composition and size-stratified lift to the size definition}} \\label{{tab:ec_size_comparison}}
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{6pt}}
\\renewcommand{{\\arraystretch}}{{1.10}}
\\begin{{tabular}}{{lcc}}
\\hline
\\multicolumn{{3}}{{l}}{{\\textit{{Panel A. Size composition of future successful entrants (main time-split test)}}}} \\\\
Size class & Employment-only & EC-style (headcount + turnover/assets) \\\\
\\hline
{rows_a}\\hline
Micro + small share (\\%) & {share_emp} & {share_ec} \\\\
\\hline
\\multicolumn{{3}}{{l}}{{\\textit{{Panel B. Top-decile lift by size group (single fixed model, regrouped)}}}} \\\\
Size group & Employment-only & EC-style (headcount + turnover/assets) \\\\
\\hline
{rows_b}\\hline
\\end{{tabular}}
\\note{{Note: A single Gradient Boosting model is trained once on the full candidate sample with the employment-only feature set, and its predictions are held fixed; only the size definition used to group firms changes across the two columns. The employment-only class is the one used as a model feature; the EC-style class follows the headcount bands of Recommendation 2003/361/EC together with the turnover-or-balance-sheet ceilings (turnover $=$ goods sales $+$ services), as a firm-level approximation that does not account for partner or linked enterprises. The main time-split test contains {n_ent} entrants; Panel A reports the {n_class} entrants with a valid size classification. Among these {n_class} classified entrants, {n_same} ({conc}\\%) receive the same class under both definitions. Panel B reports top-decile lift within each size group, with 95\\% firm-clustered bootstrap intervals. Because the model is held fixed and only regrouped, Panel B differs from Table~\\ref{{tab:size_ci}}, which instead re-estimates a separate model within each size class.}}
\\end{{table}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default="entry_from_zero_to_success_t1")
    ap.add_argument("--feature-set", default="main_t")
    ap.add_argument("--split-design", default="time")
    ap.add_argument("--model", default="gradient_boosting")
    ap.add_argument("--drop-features", default="book_leverage")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--min-test-events", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    fid, yr = cfg["columns"]["firm_id"], cfg["columns"]["year"]
    tables = resolve(cfg["paths"]["tables_dir"])
    paper_tables = tables.parent / "paper" / "tables"
    overleaf_tables = tables.parent.parent / "overleaf" / "tables"
    for d in (paper_tables, overleaf_tables):
        d.mkdir(parents=True, exist_ok=True)

    df = load_model_sample(cfg, args.outcome)
    if "size_class_ec" not in df.columns:
        raise SystemExit("size_class_ec not found in the model sample. Rebuild the panel with the v8 script 03.")
    split_col = f"split_{args.split_design}"
    if split_col not in df.columns:
        raise SystemExit(f"{split_col} not found. Run script 05 first.")

    features = clean_feature_list(get_features(args.feature_set), parse_csv_arg(args.drop_features))
    train = df[df[split_col] == "train"].copy()
    valid = df[df[split_col] == "valid"].copy()
    test = df[df[split_col] == "test"].copy()
    if train.empty or valid.empty or test.empty:
        raise SystemExit("Empty train/valid/test on the requested split.")
    model, best_params, _ = tune_on_validation(args.model, train, valid, features, args.outcome,
                                                seed=args.seed, quick=args.quick)

    score = predict_scores(model, test[features])
    tp = test[[fid, yr, "size_class", "size_class_ec", args.outcome]].copy()
    tp = tp.rename(columns={args.outcome: "y_true"})
    tp["score"] = score

    entrants = tp[tp["y_true"].astype(float) == 1.0]
    comp_emp = composition(entrants, "size_class")
    comp_ec = composition(entrants, "size_class_ec")
    cand_emp = composition(tp, "size_class")
    cand_ec = composition(tp, "size_class_ec")

    both = entrants.dropna(subset=["size_class", "size_class_ec"])
    ct = pd.crosstab(both["size_class"], both["size_class_ec"]).reindex(
        index=SIZE_ORDER, columns=SIZE_ORDER, fill_value=0)
    n_both = int(len(both))
    same = int(sum(ct.loc[k, k] for k in SIZE_ORDER))
    concord = {"n_entrants": int(len(entrants)), "n_entrants_both_classified": n_both,
               "n_same_class": same, "concordance_pct": (100.0 * same / n_both) if n_both else np.nan}

    lift_emp = group_lift(tp, "size_class", fid, args.bootstrap, args.seed, args.min_test_events)
    lift_ec = group_lift(tp, "size_class_ec", fid, args.bootstrap, args.seed, args.min_test_events)

    tag = f"{slug(args.outcome)}_{slug(args.feature_set)}_{slug(args.split_design)}_{slug(args.model)}"
    pd.DataFrame([{"definition": "employment_only", **comp_emp},
                  {"definition": "ec_turnover_or_assets", **comp_ec}]).to_csv(
        tables / f"ec_size_entrant_composition_{tag}.csv", index=False)
    pd.DataFrame([{"definition": "employment_only", **cand_emp},
                  {"definition": "ec_turnover_or_assets", **cand_ec}]).to_csv(
        tables / f"ec_size_candidate_composition_{tag}.csv", index=False)
    ct.to_csv(tables / f"ec_size_concordance_{tag}.csv")
    lift_long = []
    for defn, lk in [("employment_only", lift_emp), ("ec_turnover_or_assets", lift_ec)]:
        for sample, rec in lk.items():
            lift_long.append({"definition": defn, "sample": sample, **rec})
    pd.DataFrame(lift_long).to_csv(tables / f"ec_size_fixed_model_lift_{tag}.csv", index=False)

    latex = build_latex(comp_emp, comp_ec, concord, lift_emp, lift_ec,
                        args.outcome, args.split_design, args.model)
    (paper_tables / "tab_ec_size_comparison_v1.tex").write_text(latex)
    (overleaf_tables / "tab_ec_size_comparison_v1.tex").write_text(latex)

    print("\n=== EC vs EMPLOYMENT SIZE DEFINITION (fixed-model regrouping) ===")
    print(f"best_params (full-sample model): {best_params}")
    print(f"Entrants (time-split test): {concord['n_entrants']:,}")
    print(f"  micro+small share of entrants -- employment: {fmt(comp_emp['micro_small_share'],1)}%  |  EC: {fmt(comp_ec['micro_small_share'],1)}%")
    print(f"  concordance on entrants: {fmt(concord['concordance_pct'],1)}% ({same}/{n_both})")
    print("\nPanel B (fixed-model lift@10, regrouped):")
    for k in PANEL_B_ROWS:
        e, c = lift_emp.get(k, {}), lift_ec.get(k, {})
        print(f"  {PANEL_B_LABELS[k]:<14} emp {fmt(e.get('lift'))} [{fmt(e.get('lo'))},{fmt(e.get('hi'))}]"
              f"   |   EC {fmt(c.get('lift'))} [{fmt(c.get('lo'))},{fmt(c.get('hi'))}]")
    print("\nConcordance cross-tab (rows=employment, cols=EC):")
    print(ct.to_string())
    print(f"\nWrote CSVs to {tables}")
    print(f"Wrote appendix fragment tab_ec_size_comparison_v1.tex to {paper_tables} and {overleaf_tables}")


if __name__ == "__main__":
    main()
