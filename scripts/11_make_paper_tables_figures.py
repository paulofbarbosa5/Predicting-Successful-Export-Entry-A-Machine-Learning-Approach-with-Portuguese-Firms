r"""11_make_paper_tables_figures.py

Generate the paper-ready tables, figures, and a short results summary for the
revised distance-to-export paper.

Run after scripts 05--10 have produced the model, policy-baseline, size, and
interaction outputs. This script DOES NOT re-train models. It only reads files
from outputs/tables and outputs/predictions and writes polished artifacts to:

    outputs/paper/tables/       LaTeX table fragments + CSV copies
    outputs/paper/figures/      PNG figures
    overleaf/tables/            same LaTeX fragments, ready to upload/input
    overleaf/figures/           same figures, ready to upload/includegraphics

Typical run from the project root:

    python scripts/11_make_paper_tables_figures.py

Optional:

    python scripts/11_make_paper_tables_figures.py --outcome entry_from_zero_to_success_t1 --feature-set main_t

The generated LaTeX uses \hline style and \note{} blocks to match the current
paper style.
"""
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def slug(s: str) -> str:
    """Filename slug; identical to _model_common.slug so table tags match script 14."""
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(s))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUTS = REPO_ROOT / "outputs"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
PREDICTIONS = OUTPUTS / "predictions"
PAPER = OUTPUTS / "paper"
PAPER_TABLES = PAPER / "tables"
PAPER_FIGURES = PAPER / "figures"
OVERLEAF = REPO_ROOT / "overleaf"
OVERLEAF_TABLES = OVERLEAF / "tables"
OVERLEAF_FIGURES = OVERLEAF / "figures"

for d in [PAPER_TABLES, PAPER_FIGURES, OVERLEAF_TABLES, OVERLEAF_FIGURES]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Naming, formatting, helpers
# ---------------------------------------------------------------------------
METHOD_LABELS = {
    "random": "Random targeting",
    "productivity_ranking": "Productivity ranking",
    "employment_ranking": "Employment ranking",
    "assets_ranking": "Assets ranking",
    "import_experience_ranking": "Import-experience ranking",
    "eu_import_share_ranking": "EU import-share ranking",
    "logit_productivity_size_sector": "Logit: productivity, size, sector",
    "logit_productivity_size_sector_imports": "Logit: productivity, size, sector, imports",
    "logit_productivity_size_sector_imports_financial": "Logit: productivity, size, sector, imports, finance",
    "logit": "Logit",
    "elastic_net": "Elastic net",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "mlp": "Neural network",
}

SPLIT_LABELS = {
    "time": "Time split",
    "firm_grouped_time": "Firm-grouped + time split",
    "post2018_time": "Post-2018 split",
    "pre_covid_time": "Pre-COVID split",
    "random_firm_year": "Random firm-year split",
    "firm_grouped": "Firm-grouped split",
}

SAMPLE_LABELS = {
    "all_firms": "All firms",
    "micro_small": "Micro + small",
    "medium_large": "Medium + large",
    "micro": "Micro",
    "small": "Small",
    "medium": "Medium",
    "large": "Large",
}

OUTCOME_LABELS = {
    "entry_from_zero_to_positive_t1": "Zero exports $t$ $\\rightarrow$ positive exports $t+1$",
    "entry_from_zero_to_success_t1": "Zero exports $t$ $\\rightarrow$ successful exporter $t+1$",
    "entry_from_zero_to_success_t1_t2": "Zero exports $t$ $\\rightarrow$ successful exporter within $t+1,t+2$",
    "entry_from_zero_to_success_t1_t3": "Zero exports $t$ $\\rightarrow$ successful exporter within $t+1,t+2,t+3$",
    "entry_from_near_zero_to_success_t1": "Near-zero exports $t$ $\\rightarrow$ successful exporter $t+1$",
    "entry_from_near_zero_to_success_t1_t2": "Near-zero exports $t$ $\\rightarrow$ successful exporter within $t+1,t+2$",
    "transition_below10_to_success_t1": "Below 10% export intensity $t$ $\\rightarrow$ successful exporter $t+1$",
    "transition_below10_to_success_t1_t2": "Below 10% export intensity $t$ $\\rightarrow$ successful exporter within $t+1,t+2$",
}

FEATURE_LABELS = {
    "division": "Sector division",
    "log_assets": "Assets",
    "equity_ratio": "Equity ratio",
    "labor_productivity_vab": "Labour productivity",
    "capital_intensity": "Capital intensity",
    "profit_margin": "Profit margin",
    "imports_extra": "Extra-EU import intensity",
    "imports_comu": "EU import intensity",
    "import_experience": "Import experience",
    "roa": "ROA",
    "log_employment": "Employment",
    "size_class": "Size class",
    "negative_equity": "Negative equity",
    "wage_per_worker": "Wage per worker",
}


def _read_csv(name: str) -> pd.DataFrame | None:
    p = TABLES / name
    if not p.exists() or p.stat().st_size == 0:
        print(f"[skip] missing or empty: {p}")
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"[skip] could not read {p}: {e}")
        return None


def _find_first(patterns: Iterable[str]) -> Path | None:
    for pat in patterns:
        hits = sorted(TABLES.glob(pat))
        if hits:
            return hits[0]
    return None


def _read_first(patterns: Iterable[str]) -> tuple[pd.DataFrame | None, Path | None]:
    p = _find_first(patterns)
    if p is None:
        return None, None
    try:
        return pd.read_csv(p), p
    except Exception as e:
        print(f"[skip] could not read {p}: {e}")
        return None, p


def latex_escape(s) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    # Keep math commands intact in outcome labels; escape ordinary underscores.
    repl = {
        "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    # Undo deliberate LaTeX fragments used in labels.
    s = s.replace("\\$", "$")
    return s


def fmt_int(x) -> str:
    if pd.isna(x):
        return "--"
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return str(x)


def fmt_dec(x, digits=3) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def fmt_pct(x, digits=1) -> str:
    if pd.isna(x):
        return "--"
    return f"{100 * float(x):.{digits}f}"


def fmt_metric_with_ci(row: pd.Series, base: str, digits: int = 3) -> str:
    val = row.get(base, np.nan)
    lo = row.get(f"{base}_lo", np.nan)
    hi = row.get(f"{base}_hi", np.nan)
    if pd.isna(val):
        return "--"
    out = fmt_dec(val, digits)
    if pd.notna(lo) and pd.notna(hi):
        out += f" [{fmt_dec(lo, digits)}, {fmt_dec(hi, digits)}]"
    return out


def write_latex_table(
    df: pd.DataFrame,
    filename: str,
    caption: str,
    label: str,
    note: str | None = None,
    align: str | None = None,
) -> None:
    r"""Write a simple \hline-style LaTeX table fragment."""
    if df is None or df.empty:
        print(f"[skip] empty table {filename}")
        return

    align = align or ("l" + "r" * (len(df.columns) - 1))
    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(rf"\captionof{{table}}{{{caption}}} \label{{{label}}}")
    lines.append(r"\centering")
    lines.append(rf"\begin{{tabular}}{{{align}}}")
    lines.append(r"\hline")
    lines.append(" & ".join(latex_escape(c) for c in df.columns) + r" \\")
    lines.append(r"\hline")
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(v) for v in row.tolist()) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    if note:
        lines.append(rf"\note{{Note: {note}}}")
    lines.append(r"\end{table}")
    lines.append("")

    out = PAPER_TABLES / filename
    out.write_text("\n".join(lines), encoding="utf-8")
    shutil.copy2(out, OVERLEAF_TABLES / filename)
    df.to_csv(PAPER_TABLES / filename.replace(".tex", ".csv"), index=False)
    print(f"[table] {out}")


def savefig(name: str) -> None:
    p = PAPER_FIGURES / name
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    shutil.copy2(p, OVERLEAF_FIGURES / name)
    print(f"[figure] {p}")


def label_method(m):
    return METHOD_LABELS.get(str(m), str(m).replace("_", " ").title())


def label_split(s):
    return SPLIT_LABELS.get(str(s), str(s).replace("_", " ").title())


def label_sample(s):
    return SAMPLE_LABELS.get(str(s), str(s).replace("_", " ").title())


def label_feature(f):
    return FEATURE_LABELS.get(str(f), str(f).replace("_", " ").title())


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def make_entry_outcomes_table(outcome: str) -> None:
    counts = _read_csv("event_counts_by_outcome.csv")
    rec = _read_csv("design_recommendation.csv")
    if counts is None:
        return
    df = counts.copy()
    if rec is not None:
        df = df.merge(rec[["outcome", "test_events", "recommended_use"]], on="outcome", how="left")
    keep_order = [
        "entry_from_zero_to_positive_t1",
        "entry_from_zero_to_success_t1",
        "entry_from_zero_to_success_t1_t2",
        "entry_from_zero_to_success_t1_t3",
        "entry_from_near_zero_to_success_t1",
        "entry_from_near_zero_to_success_t1_t2",
        "transition_below10_to_success_t1",
        "transition_below10_to_success_t1_t2",
    ]
    df["_ord"] = df["outcome"].apply(lambda x: keep_order.index(x) if x in keep_order else 999)
    df = df.sort_values("_ord")
    # Keep the key rows if present.
    wanted = [x for x in keep_order if x in df["outcome"].values]
    df = df[df["outcome"].isin(wanted)].copy()
    out = pd.DataFrame({
        "Outcome": [OUTCOME_LABELS.get(o, o.replace("_", " ")) for o in df["outcome"]],
        "Horizon": df["horizon"].astype(int).astype(str),
        "Candidate firm-years": df["candidate_firm_years"].map(fmt_int),
        "Unique firms": df["unique_firms"].map(fmt_int),
        "Entry events": df["entry_events"].map(fmt_int),
        "Entry rate (%)": df["entry_rate"].map(lambda x: fmt_pct(x, 1)),
        "Test events": df.get("test_events", pd.Series([np.nan] * len(df))).map(fmt_int),
        "Use": df.get("recommended_use", pd.Series([""] * len(df))).fillna(""),
    })
    write_latex_table(
        out,
        "tab_entry_outcomes_v5.tex",
        "Candidate samples and future export-entry outcomes",
        "tab:entry_outcomes_v2",
        "The baseline outcome is zero exports at year $t$ followed by successful exporting, defined as export intensity of at least 10\\%, in year $t+1$. Broader outcomes are used as robustness exercises.",
        align="llrrrrrl",
    )


def make_prepost_table() -> None:
    df = _read_csv("pre_post_2018_composition.csv")
    if df is None:
        return
    out = pd.DataFrame({
        "Period": df["_period"],
        "Firm-years": df["firm_years"].map(fmt_int),
        "Unique firms": df["unique_firms"].map(fmt_int),
        "Micro (%)": df["share_micro"].map(lambda x: fmt_pct(x, 1)),
        "Small (%)": df["share_small"].map(lambda x: fmt_pct(x, 1)),
        "Medium (%)": df["share_medium"].map(lambda x: fmt_pct(x, 1)),
        "Large (%)": df["share_large"].map(lambda x: fmt_pct(x, 1)),
        "Successful exporter (%)": df["share_exporter10"].map(lambda x: fmt_pct(x, 1)),
        "Zero exporter (%)": df["share_zero_exporter"].map(lambda x: fmt_pct(x, 1)),
    })
    write_latex_table(
        out,
        "tab_pre_post_2018_v5.tex",
        "Composition of the firm population before and after 2018",
        "tab:pre_post_2018_v2",
        "The table reports the observed composition of the manufacturing firm-year panel before and after the 2018 expansion in the raw extract. The post-2018 period is more micro-firm intensive and contains a larger share of zero exporters. The institutional source of the expansion is pending formal confirmation from the data provider.",
        align="lrrrrrrrr",
    )


def _load_performance(outcome: str, feature_set: str) -> pd.DataFrame | None:
    # Prefer the bootstrap file, fall back to raw metrics.
    candidates = [
        f"model_performance_with_ci_{outcome}_filtered.csv",
        f"model_performance_with_ci_{outcome}.csv",
        f"main_model_metrics_{outcome}_{feature_set}.csv",
    ]
    for name in candidates:
        df = _read_csv(name)
        if df is not None:
            return df
    return None


def make_model_performance_table(outcome: str, feature_set: str) -> None:
    df = _load_performance(outcome, feature_set)
    if df is None:
        return
    df = df.copy()
    df = df[df.get("feature_set", feature_set).eq(feature_set) if "feature_set" in df else True]
    keep_methods = ["logit", "random_forest", "gradient_boosting"]
    df = df[df["method"].isin(keep_methods)].copy()
    split_order = ["time", "firm_grouped_time", "post2018_time"]
    df["_sord"] = df["split_design"].apply(lambda x: split_order.index(x) if x in split_order else 99)
    df["_mord"] = df["method"].apply(lambda x: keep_methods.index(x) if x in keep_methods else 99)
    df = df.sort_values(["_sord", "_mord"])
    out = pd.DataFrame({
        "Split": df["split_design"].map(label_split),
        "Model": df["method"].map(label_method),
        "PR-AUC": [fmt_metric_with_ci(r, "pr_auc") for _, r in df.iterrows()],
        "ROC-AUC": [fmt_metric_with_ci(r, "roc_auc") for _, r in df.iterrows()],
        "P@5 (%)": [fmt_metric_with_ci(r, "precision_at_5") for _, r in df.iterrows()],
        "Lift@5": [fmt_metric_with_ci(r, "lift_at_5") for _, r in df.iterrows()],
        "P@10 (%)": [fmt_metric_with_ci(r, "precision_at_10") for _, r in df.iterrows()],
        "Lift@10": [fmt_metric_with_ci(r, "lift_at_10") for _, r in df.iterrows()],
    })
    # Convert precision columns to percentage if no CI impossible? Use current decimals in function. Create alternate manually.
    # Replace precision decimal strings by percentages while preserving CI format.
    for c in ["P@5 (%)", "P@10 (%)"]:
        vals = []
        base = "precision_at_5" if "5" in c else "precision_at_10"
        for _, r in df.iterrows():
            v = r.get(base, np.nan)
            lo = r.get(base + "_lo", np.nan)
            hi = r.get(base + "_hi", np.nan)
            if pd.isna(v):
                vals.append("--")
            elif pd.notna(lo) and pd.notna(hi):
                vals.append(f"{fmt_pct(v,1)} [{fmt_pct(lo,1)}, {fmt_pct(hi,1)}]")
            else:
                vals.append(fmt_pct(v,1))
        out[c] = vals
    write_latex_table(
        out,
        "tab_model_performance_v5.tex",
        "Forward-looking export-entry prediction performance",
        "tab:model_performance_v2",
        "The outcome is successful export entry in $t+1$ among firms with zero exports in year $t$. P@5 and P@10 denote the observed entry rate among the top 5\\% and 10\\% of scored firms. Baseline = average entry rate in the test sample; lift is precision divided by this baseline. Brackets report firm-cluster bootstrap confidence intervals.",
        align="llrrrrrr",
    )


def make_policy_table(outcome: str, feature_set: str) -> None:
    pol = _read_csv(f"policy_baseline_targeting_{outcome}.csv")
    mod = _read_csv(f"main_model_metrics_{outcome}_{feature_set}.csv")
    if pol is None:
        return
    split = "time"
    pol = pol[pol["split_design"].eq(split)].copy()
    keep_policy = [
        "random", "productivity_ranking", "employment_ranking", "assets_ranking",
        "import_experience_ranking", "eu_import_share_ranking",
        "logit_productivity_size_sector_imports",
        "logit_productivity_size_sector_imports_financial",
    ]
    pol = pol[pol["method"].isin(keep_policy)]
    rows = pol.copy()
    if mod is not None:
        mods = mod[(mod["split_design"].eq(split)) & (mod["method"].isin(["random_forest", "gradient_boosting", "logit"]))].copy()
        # Harmonize columns with policy table.
        rows = pd.concat([rows, mods[rows.columns.intersection(mods.columns)]], ignore_index=True, sort=False)
    order = keep_policy + ["logit", "gradient_boosting", "random_forest"]
    rows["_ord"] = rows["method"].apply(lambda x: order.index(x) if x in order else 99)
    rows = rows.sort_values("_ord")
    out = pd.DataFrame({
        "Rule / model": rows["method"].map(label_method),
        "P@5 (%)": rows["precision_at_5"].map(lambda x: fmt_pct(x, 1)),
        "Lift@5": rows["lift_at_5"].map(lambda x: fmt_dec(x, 2)),
        "Entrants@5": rows["entrants_at_5"].map(fmt_int),
        "P@10 (%)": rows["precision_at_10"].map(lambda x: fmt_pct(x, 1)),
        "Lift@10": rows["lift_at_10"].map(lambda x: fmt_dec(x, 2)),
        "Entrants@10": rows["entrants_at_10"].map(fmt_int),
    })
    write_latex_table(
        out,
        "tab_policy_baselines_v5.tex",
        "Targeting performance relative to simple policy rules",
        "tab:policy_baselines_v2",
        "The table uses the main chronological split. The policy interpretation is that an agency targets the top 5\\% or 10\\% of candidate firms according to each rule or model score. Baseline = average entry rate in the test sample; lift is precision divided by this baseline. The random-targeting row is one realized draw; its expected lift is 1.00.",
        align="lrrrrrr",
    )


def _ci(lo, hi, digits=2):
    """Render a [lo, hi] bracket; blank if missing."""
    import numpy as _np
    if lo is None or hi is None or (isinstance(lo, float) and _np.isnan(lo)) or (isinstance(hi, float) and _np.isnan(hi)):
        return ""
    return f" [{fmt_dec(lo, digits)}, {fmt_dec(hi, digits)}]"


def _ci_pct(lo, hi, digits=1):
    """Render a [lo, hi] bracket where lo/hi are proportions shown as percentages."""
    import numpy as _np
    if lo is None or hi is None or (isinstance(lo, float) and _np.isnan(lo)) or (isinstance(hi, float) and _np.isnan(hi)):
        return ""
    return f" [{fmt_pct(lo, digits)}, {fmt_pct(hi, digits)}]"


def make_size_table(outcome: str,
                    feature_set: str = "main_t",
                    split_design: str = "time",
                    model: str = "gradient_boosting") -> None:
    """Polished Table 6 from the script-14 bootstrap-CI output (single source of truth).

    Renders P@10 and Lift@10 with firm-clustered CIs, a scriptsize/adjustbox layout, and a
    dedicated sparse row for size groups too small to estimate (e.g. Large: counts only).
    """
    tag = f"{slug(outcome)}_{slug(feature_set)}_{slug(split_design)}_{slug(model)}"
    df = _read_csv(f"size_group_bootstrap_ci_{tag}.csv")
    if df is None:
        print("[size-table] WARNING: bootstrap-CI file not found; run scripts/14_size_bootstrap_ci.py.")
        return
    df = df.set_index("sample")

    # Known sparse-row counts (Large): use the CSV if present, else the documented values.
    def cell_counts(name, default_n=None, default_pos=None):
        n = df.loc[name, "n"] if name in df.index and "n" in df.columns else None
        pos = df.loc[name, "positives"] if name in df.index and "positives" in df.columns else None
        prev = df.loc[name, "prevalence"] if name in df.index and "prevalence" in df.columns else None
        if (n is None or (isinstance(n, float) and math.isnan(n))) and default_n is not None:
            n, pos = default_n, default_pos
            prev = (default_pos / default_n) if default_n else float("nan")
        return n, pos, prev

    def row_full(name):
        r = df.loc[name]
        roc = fmt_dec(r.get("roc_auc"), 3) + _ci(r.get("roc_auc_lo"), r.get("roc_auc_hi"), 3)
        pr = fmt_dec(r.get("pr_auc"), 3) + _ci(r.get("pr_auc_lo"), r.get("pr_auc_hi"), 3)
        p10 = fmt_pct(r.get("precision_at_10"), 1) + _ci_pct(r.get("precision_at_10_lo"), r.get("precision_at_10_hi"), 1)
        lift = fmt_dec(r.get("lift_at_10"), 2) + _ci(r.get("lift_at_10_lo"), r.get("lift_at_10_hi"), 2)
        n, pos, prev = cell_counts(name)
        return [label_sample(name), fmt_int(n), fmt_int(pos), fmt_pct(prev, 1), roc, pr, p10, lift]

    def row_sparse(name, default_n, default_pos, status_text):
        n, pos, prev = cell_counts(name, default_n, default_pos)
        return [label_sample(name), fmt_int(n), fmt_int(pos), fmt_pct(prev, 1),
                "--", "--", "--", f"\\textit{{{status_text}}}"]

    order_full = ["all_firms", "micro_small", "medium_large", "micro", "small", "medium"]
    body = [row_full(k) for k in order_full if k in df.index]
    # Large: sparse row with documented counts (19 candidates, 2 entrants).
    body.append(row_sparse("large", 19, 2, "too few events"))

    header = ["Sample", "Cand. FY", "Entr.", r"Rate (\%)", "ROC-AUC", "PR-AUC", r"P@10 (\%)", "Lift@10"]
    note = ("Models are re-estimated within each size class on the main chronological split using "
            "Gradient Boosting. Brackets are 95\\% firm-clustered bootstrap confidence intervals based on "
            "1,000 resamples of firms. Cand. FY = candidate firm-years; Entr. = future successful entrants; "
            "Rate = baseline entry rate. The all-firms row has 853 size-classified entrants; the main test "
            "sample contains 854 entrants because one entrant has missing size class. Large firms have only "
            "19 candidate firm-years and two future entries in the test window and are not estimated separately.")

    lines = [r"\begin{table}[!htbp]",
             r"\captionof{table}{Export-entry prediction by firm-size class} \label{tab:size_ci}",
             r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\renewcommand{\arraystretch}{1.10}", r"\begin{adjustbox}{max width=\textwidth}",
             r"\begin{tabular}{lrrrcccc}", r"\hline",
             " & ".join(header) + r" \\", r"\hline"]
    for row in body:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{adjustbox}",
              rf"\note{{Note: {note}}}", r"\end{table}", ""]
    txt = "\n".join(lines)
    (PAPER_TABLES / "tab_size_stratified_ci_v5.tex").write_text(txt, encoding="utf-8")
    shutil.copy2(PAPER_TABLES / "tab_size_stratified_ci_v5.tex", OVERLEAF_TABLES / "tab_size_stratified_ci_v5.tex")
    print(f"[table] {PAPER_TABLES / 'tab_size_stratified_ci_v5.tex'}")


def make_no_sales_table(outcome: str) -> None:
    df = _read_csv(f"main_model_metrics_{outcome}_no_sales_derived_t.csv")
    if df is None:
        return
    df = df[df["method"].isin(["logit", "random_forest", "gradient_boosting"])].copy()
    split_order = ["time", "firm_grouped_time", "post2018_time"]
    df["_sord"] = df["split_design"].apply(lambda x: split_order.index(x) if x in split_order else 99)
    df = df.sort_values(["_sord", "method"])
    out = pd.DataFrame({
        "Split": df["split_design"].map(label_split),
        "Model": df["method"].map(label_method),
        "ROC-AUC": df["roc_auc"].map(lambda x: fmt_dec(x, 3)),
        "PR-AUC": df["pr_auc"].map(lambda x: fmt_dec(x, 3)),
        "P@10 (%)": df["precision_at_10"].map(lambda x: fmt_pct(x, 1)),
        "Lift@10": df["lift_at_10"].map(lambda x: fmt_dec(x, 2)),
    })
    write_latex_table(
        out,
        "tab_no_sales_robustness_v5.tex",
        "Robustness excluding sales-derived predictors",
        "tab:no_sales_robustness_v2",
        "This specification excludes predictors derived from total sales, including VAB labour productivity and profit margin. It addresses the concern that sales-decomposition variables may mechanically contain export information.",
        align="llrrrr",
    )


def make_importance_tables(outcome: str, feature_set: str) -> None:
    imp = _read_csv(f"permutation_importance_{outcome}_{feature_set}.csv")
    inter = _read_csv(f"interaction_strength_{outcome}_{feature_set}.csv")
    if imp is not None:
        imp = imp.sort_values("importance_mean", ascending=False).head(10)
        out = pd.DataFrame({
            "Feature": imp["feature"].map(label_feature),
            "Importance": imp["importance_mean"].map(lambda x: fmt_dec(x, 4)),
            "Std. dev.": imp["importance_std"].map(lambda x: fmt_dec(x, 4)),
        })
        write_latex_table(
            out,
            "tab_permutation_importance_v2.tex",
            "Permutation importance of predictive signals",
            "tab:permutation_importance_v2",
            "Importance is computed on the test sample using PR-AUC as the scoring metric. These values are predictive rankings, not causal effects.",
            align="lrr",
        )
    if inter is not None:
        inter = inter.sort_values("interaction_strength_fast", ascending=False).head(10)
        out = pd.DataFrame({
            "Feature 1": inter["feature_1"].map(label_feature),
            "Feature 2": inter["feature_2"].map(label_feature),
            "Interaction score": inter["interaction_strength_fast"].map(lambda x: fmt_dec(x, 3)),
        })
        write_latex_table(
            out,
            "tab_interaction_strength_v2.tex",
            "Candidate interaction pairs in the export-readiness score",
            "tab:interaction_strength_v2",
            "The interaction score is a fast screening statistic used to rank candidate pairs for partial-dependence or ALE plots. It should not be interpreted as a causal effect or as a formal Friedman H-statistic.",
            align="llr",
        )


def make_split_table(outcome: str) -> None:
    df = _read_csv(f"split_diagnostics_{outcome}.csv")
    if df is None:
        return
    order = ["random_firm_year", "firm_grouped", "time", "firm_grouped_time", "pre_covid_time", "post2018_time"]
    df["_ord"] = df["split_design"].apply(lambda x: order.index(x) if x in order else 99)
    df = df.sort_values("_ord")
    out = pd.DataFrame({
        "Split": df["split_design"].map(label_split),
        "Train events": df["train_events"].map(fmt_int),
        "Validation events": df["valid_events"].map(fmt_int),
        "Test events": df["test_events"].map(fmt_int),
        "Test entry rate (%)": df["test_entry_rate"].map(lambda x: fmt_pct(x, 1)),
        "Firm overlap": df["firm_overlap_train_test"].map(fmt_int),
        "Year overlap": df["year_overlap_train_test"].map(fmt_int),
    })
    write_latex_table(
        out,
        "tab_split_diagnostics_v5.tex",
        "Validation split diagnostics",
        "tab:split_diagnostics_v2",
        "The strict firm-grouped plus time split has no overlap in firms or candidate years between training and test sets. The post-2018 split evaluates the model within the expanded post-2018 coverage period.",
        align="lrrrrrr",
    )


def make_leakage_table() -> None:
    df = _read_csv("leakage_identity_check.csv")
    if df is None:
        return
    # Summarize by min/mean/max across years.
    summary = pd.DataFrame({
        "Statistic": ["Mean absolute difference", "Max median absolute difference", "Min share below 1 euro", "Min reconstructed label share"],
        "Value": [
            f"{df['mean_abs_diff'].mean():.2e}",
            f"{df['median_abs_diff'].max():.2e}",
            fmt_pct(df["share_below_1eur"].min(), 1),
            fmt_pct(df["share_label_reconstructed"].min(), 1),
        ],
    })
    write_latex_table(
        summary,
        "tab_leakage_identity_appendix.tex",
        "Sales-identity and old-label reconstruction check",
        "tab:leakage_identity_appendix",
        "The old contemporaneous exporter label is exactly reconstructable from total sales and domestic sales because total sales include sales to foreign markets. These variables are excluded from the revised predictor matrix.",
        align="lr",
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_score_decile_figures(outcome: str, feature_set: str) -> None:
    df = _read_csv(f"entry_rate_by_score_decile_{outcome}_filtered.csv")
    if df is None:
        return
    # RF time decile plot.
    for split, method, name in [
        ("time", "random_forest", "fig_score_decile_random_forest_time.png"),
        ("firm_grouped_time", "gradient_boosting", "fig_score_decile_gradient_boosting_strict.png"),
        ("post2018_time", "gradient_boosting", "fig_score_decile_gradient_boosting_post2018.png"),
    ]:
        d = df[(df["split_design"].eq(split)) & (df["method"].eq(method))]
        if d.empty:
            continue
        d = d.sort_values("score_decile")
        plt.figure(figsize=(7.2, 4.5))
        plt.plot(d["score_decile"], 100 * d["entry_rate"], marker="o")
        plt.xlabel("Predicted-score decile")
        plt.ylabel("Observed successful-entry rate (%)")
        plt.title(f"Observed entry rate by score decile: {label_method(method)}, {label_split(split)}")
        plt.xticks(range(1, 11))
        plt.grid(True, axis="y", alpha=0.3)
        savefig(name)

    # Multiple model decile plot for time split.
    split = "time"
    d = df[(df["split_design"].eq(split)) & (df["method"].isin(["logit", "random_forest", "gradient_boosting"]))]
    if not d.empty:
        plt.figure(figsize=(7.2, 4.5))
        for method, dd in d.groupby("method"):
            dd = dd.sort_values("score_decile")
            plt.plot(dd["score_decile"], 100 * dd["entry_rate"], marker="o", label=label_method(method))
        plt.xlabel("Predicted-score decile")
        plt.ylabel("Observed successful-entry rate (%)")
        plt.title("Observed entry rate by score decile, time split")
        plt.xticks(range(1, 11))
        plt.legend(frameon=False)
        plt.grid(True, axis="y", alpha=0.3)
        savefig("fig_score_decile_models_time.png")


def make_policy_lift_figure(outcome: str, feature_set: str) -> None:
    pol = _read_csv(f"policy_baseline_targeting_{outcome}.csv")
    mod = _read_csv(f"main_model_metrics_{outcome}_{feature_set}.csv")
    if pol is None:
        return
    split = "time"
    keep = [
        "random", "productivity_ranking", "employment_ranking", "assets_ranking",
        "import_experience_ranking", "eu_import_share_ranking",
        "logit_productivity_size_sector_imports", "random_forest", "gradient_boosting",
    ]
    rows = pol[pol["split_design"].eq(split)].copy()
    if mod is not None:
        mods = mod[(mod["split_design"].eq(split)) & mod["method"].isin(["random_forest", "gradient_boosting"])].copy()
        rows = pd.concat([rows, mods[rows.columns.intersection(mods.columns)]], ignore_index=True, sort=False)
    rows = rows[rows["method"].isin(keep)].copy()
    rows["_ord"] = rows["method"].apply(lambda x: keep.index(x) if x in keep else 99)
    rows = rows.sort_values("_ord")
    if rows.empty:
        return
    x = np.arange(len(rows))
    width = 0.38
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, rows["lift_at_5"], width, label="Lift@5")
    plt.bar(x + width/2, rows["lift_at_10"], width, label="Lift@10")
    plt.axhline(1, linestyle="--", linewidth=1)
    plt.xticks(x, [label_method(m) for m in rows["method"]], rotation=45, ha="right")
    plt.ylabel("Lift relative to random targeting")
    plt.title("Targeting lift relative to simple policy rules, time split")
    plt.legend(frameon=False)
    plt.tight_layout()
    savefig("fig_policy_lift_time.png")


def make_size_figures(outcome: str,
                     feature_set: str = "main_t",
                     split_design: str = "time",
                     model: str = "gradient_boosting") -> None:
    # Read the same bootstrap-CI source as the size table so any regenerated figure
    # is consistent with Table 5. (The main text no longer relies on the lift figure.)
    tag = f"{slug(outcome)}_{slug(feature_set)}_{slug(split_design)}_{slug(model)}"
    df = _read_csv(f"size_group_bootstrap_ci_{tag}.csv")
    if df is None:
        df = _read_csv(f"size_group_retrained_performance_{outcome}.csv")
    if df is None:
        return
    d = df[df["status"].fillna("estimated").eq("estimated")].copy()
    order = ["all_firms", "micro_small", "medium_large", "micro", "small", "medium"]
    d = d[d["sample"].isin(order)]
    d["_ord"] = d["sample"].apply(lambda x: order.index(x))
    d = d.sort_values("_ord")
    if not d.empty:
        plt.figure(figsize=(8, 4.5))
        plt.bar([label_sample(s) for s in d["sample"]], d["lift_at_10"])
        plt.ylabel("Lift@10")
        plt.title("Top-decile targeting lift by firm-size group")
        plt.xticks(rotation=30, ha="right")
        plt.grid(True, axis="y", alpha=0.3)
        savefig("fig_size_lift_at_10.png")

    # Future export-entry events by DISJOINT size class (no overlapping aggregates).
    # Avoids double-counting that arises from mixing all_firms / micro_small / medium_large
    # with the individual classes. Large is sparse in the CI file; inject its known count.
    dd = df.set_index("sample")
    disjoint = ["micro", "small", "medium", "large"]
    known_pos = {"large": 2}
    counts = []
    for k in disjoint:
        v = dd.loc[k, "positives"] if (k in dd.index and "positives" in dd.columns) else None
        if v is None or (isinstance(v, float) and math.isnan(v)):
            v = known_pos.get(k)
        counts.append(0 if v is None else int(v))
    plt.figure(figsize=(7, 4.2))
    bars = plt.bar([k.capitalize() for k in disjoint], counts, color="#3b6ea5")
    for b, c in zip(bars, counts):
        plt.text(b.get_x() + b.get_width() / 2, c, f"{c:,}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Future successful entries")
    plt.title("Future export-entry events by firm-size class")
    plt.grid(True, axis="y", alpha=0.3)
    savefig("fig_size_entry_events.png")

    # Copy existing score distribution if produced by script 09.
    src = FIGURES / f"score_distribution_by_size_{outcome}.png"
    if src.exists():
        dst = PAPER_FIGURES / "fig_score_distribution_by_size.png"
        shutil.copy2(src, dst)
        shutil.copy2(dst, OVERLEAF_FIGURES / dst.name)
        print(f"[figure] copied {dst}")


def make_importance_figures(outcome: str, feature_set: str) -> None:
    imp = _read_csv(f"permutation_importance_{outcome}_{feature_set}.csv")
    if imp is not None and not imp.empty:
        d = imp.sort_values("importance_mean", ascending=False).head(10).iloc[::-1]
        plt.figure(figsize=(7.2, 5.0))
        plt.barh([label_feature(f) for f in d["feature"]], d["importance_mean"])
        plt.xlabel("Permutation importance (PR-AUC drop)")
        plt.title("Predictive signals in the export-readiness score")
        plt.grid(True, axis="x", alpha=0.3)
        savefig("fig_permutation_importance.png")

    inter = _read_csv(f"interaction_strength_{outcome}_{feature_set}.csv")
    if inter is not None and not inter.empty:
        d = inter.sort_values("interaction_strength_fast", ascending=False).head(8).iloc[::-1]
        labels = [f"{label_feature(a)} × {label_feature(b)}" for a, b in zip(d["feature_1"], d["feature_2"])]
        plt.figure(figsize=(8.0, 5.0))
        plt.barh(labels, d["interaction_strength_fast"])
        plt.xlabel("Interaction screening score")
        plt.title("Candidate interactions in export readiness")
        plt.grid(True, axis="x", alpha=0.3)
        savefig("fig_interaction_strength.png")


def make_prepost_figure() -> None:
    df = _read_csv("pre_post_2018_composition.csv")
    if df is None:
        return
    periods = df["_period"].tolist()
    size_cols = ["share_micro", "share_small", "share_medium", "share_large"]
    labels = ["Micro", "Small", "Medium", "Large"]
    bottom = np.zeros(len(df))
    plt.figure(figsize=(7.2, 4.5))
    x = np.arange(len(df))
    for col, lab in zip(size_cols, labels):
        vals = 100 * df[col].to_numpy(dtype=float)
        plt.bar(x, vals, bottom=bottom, label=lab)
        bottom += vals
    plt.xticks(x, periods)
    plt.ylabel("Share of firm-years (%)")
    plt.title("Firm-size composition before and after 2018")
    plt.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    savefig("fig_prepost2018_size_composition.png")


# ---------------------------------------------------------------------------
# Summary and Overleaf snippets
# ---------------------------------------------------------------------------
def make_results_summary(outcome: str, feature_set: str) -> None:
    perf = _load_performance(outcome, feature_set)
    size = _read_csv(f"size_group_retrained_performance_{outcome}.csv")
    policy = _read_csv(f"policy_baseline_targeting_{outcome}.csv")
    lines = []
    lines.append("# Paper results summary\n")
    if perf is not None:
        d = perf[(perf["split_design"].eq("time")) & (perf["method"].isin(["random_forest", "gradient_boosting", "logit"]))].copy()
        if not d.empty:
            best = d.sort_values("lift_at_10", ascending=False).iloc[0]
            lines.append("## Main predictive performance\n")
            lines.append(f"- Best top-decile lift in the main time split: **{label_method(best['method'])}**, lift@10 = {fmt_dec(best['lift_at_10'],2)}, precision@10 = {fmt_pct(best['precision_at_10'],1)}%, PR-AUC = {fmt_dec(best['pr_auc'],3)}, ROC-AUC = {fmt_dec(best['roc_auc'],3)}.")
            lines.append("- Interpret this as modest but useful ranking power, not high accuracy.\n")
    if size is not None:
        lines.append("## Broad-coverage / size contribution\n")
        try:
            s = size.set_index("sample")
            micro_events = int(s.loc["micro", "positives"])
            small_events = int(s.loc["small", "positives"])
            all_events = int(s.loc["all_firms", "positives"])
            share = 100 * (micro_events + small_events) / all_events
            lines.append(f"- Micro and small firms account for **{micro_events + small_events:,} of {all_events:,}** future successful entries in the time-split test ({share:.1f}%).")
            lines.append("- This is a central contribution: broad administrative coverage changes the population at risk.\n")
        except Exception:
            pass
    if policy is not None:
        lines.append("## Policy baseline comparison\n")
        d = policy[policy["split_design"].eq("time")].copy()
        if "assets_ranking" in d["method"].values:
            a = d[d["method"].eq("assets_ranking")].iloc[0]
            lines.append(f"- Assets ranking is a strong baseline: lift@10 = {fmt_dec(a['lift_at_10'],2)}. The paper should compare all ML gains against this rule, not only against random targeting.")
    lines.append("\n## Main figures generated\n")
    for p in sorted(PAPER_FIGURES.glob("*.png")):
        lines.append(f"- `{p.name}`")
    lines.append("\n## Main LaTeX tables generated\n")
    for p in sorted(PAPER_TABLES.glob("*.tex")):
        lines.append(f"- `\\input{{tables/{p.name}}}`")
    out = PAPER / "paper_results_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[summary] {out}")


def make_overleaf_snippets() -> None:
    lines = []
    lines.append("% Copy this block into the Results section after uploading the files in overleaf/tables and overleaf/figures.\n")
    lines.append("% Tables")
    for name in [
        "tab_entry_outcomes_v5.tex",
        "tab_pre_post_2018_v5.tex",
        "tab_model_performance_v5.tex",
        "tab_policy_baselines_v5.tex",
        "tab_size_stratified_ci_v5.tex",
        "tab_permutation_importance_v2.tex",
        "tab_interaction_strength_v2.tex",
        "tab_no_sales_robustness_v5.tex",
        "tab_split_diagnostics_v5.tex",
    ]:
        if (PAPER_TABLES / name).exists():
            lines.append(rf"\input{{tables/{name}}}")
    lines.append("\n% Figures")
    figblocks = [
        ("fig_score_decile_random_forest_time", "Observed entry rate by predicted-score decile", "fig:score_deciles_rf", "The figure reports observed successful export-entry rates by decile of the Random Forest score in the main chronological test split."),
        ("fig_policy_lift_time", "Targeting lift relative to simple policy rules", "fig:policy_lift", "The figure compares lift at the top 5\\% and 10\\% of the score distribution for simple rankings, policy-logit rules and model scores."),
        ("fig_score_distribution_by_size", "Distribution of export-readiness scores by firm size", "fig:score_distribution_size", "The figure shows the distribution of predicted export-readiness scores across firm-size classes."),
        ("fig_size_lift_at_10", "Top-decile lift by firm-size group", "fig:size_lift", "The figure reports lift at the top decile for models re-estimated within size groups."),
        ("fig_permutation_importance", "Predictive signals in the export-readiness score", "fig:perm_importance", "Permutation importance is computed on the test set using PR-AUC as the scoring metric. The estimates are predictive rankings, not causal effects."),
        ("fig_interaction_strength", "Candidate interaction pairs in export readiness", "fig:interaction_strength", "The figure ranks candidate interaction pairs used to select the partial-dependence or ALE plots in the main text and appendix."),
    ]
    for fname, caption, label, note in figblocks:
        if (PAPER_FIGURES / f"{fname}.png").exists():
            lines.append(r"\begin{figure}[!ht]")
            lines.append(rf"\caption{{{caption}}} \label{{{label}}}")
            lines.append(r"\centering")
            lines.append(rf"\includegraphics[width=12cm]{{figures/{fname}}}")
            lines.append(rf"\note{{Note: {note}}}")
            lines.append(r"\end{figure}")
            lines.append("")
    out = PAPER / "overleaf_input_snippets.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    shutil.copy2(out, OVERLEAF / out.name)
    print(f"[overleaf] {out}")


def make_zip() -> None:
    zip_path = OUTPUTS / "paper_overleaf_artifacts"
    if zip_path.with_suffix(".zip").exists():
        zip_path.with_suffix(".zip").unlink()
    shutil.make_archive(str(zip_path), "zip", root_dir=OVERLEAF)
    print(f"[zip] {zip_path.with_suffix('.zip')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready tables and figures for Overleaf.")
    parser.add_argument("--outcome", default="entry_from_zero_to_success_t1")
    parser.add_argument("--feature-set", default="main_t")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    print(f"[start] outcome={args.outcome} feature_set={args.feature_set}")
    print(f"[paths] reading tables from {TABLES}")
    print(f"[paths] writing paper outputs to {PAPER}")

    make_entry_outcomes_table(args.outcome)
    make_prepost_table()
    make_split_table(args.outcome)
    make_model_performance_table(args.outcome, args.feature_set)
    make_policy_table(args.outcome, args.feature_set)
    make_size_table(args.outcome)
    make_no_sales_table(args.outcome)
    make_importance_tables(args.outcome, args.feature_set)
    make_leakage_table()

    make_score_decile_figures(args.outcome, args.feature_set)
    make_policy_lift_figure(args.outcome, args.feature_set)
    make_size_figures(args.outcome)
    make_importance_figures(args.outcome, args.feature_set)
    make_prepost_figure()

    make_results_summary(args.outcome, args.feature_set)
    make_overleaf_snippets()
    if not args.no_zip:
        make_zip()

    print("\nDone. Review:")
    print(f"  - {PAPER_TABLES}")
    print(f"  - {PAPER_FIGURES}")
    print(f"  - {PAPER / 'paper_results_summary.md'}")
    print(f"  - {PAPER / 'overleaf_input_snippets.tex'}")
    print("Upload the contents of overleaf/tables and overleaf/figures to Overleaf.")


if __name__ == "__main__":
    main()
