"""Compute real SHAP for the forward-looking GB model (main_t, time split).

Reproduces the pipeline's gradient-boosting model used for the importance/
interaction tables, then computes:
  (1) global mean|SHAP| per ORIGINAL feature (one-hot cols summed back),
  (2) SHAP dependence for labour productivity, split by import_experience,
  (3) SHAP interaction values on a sample -> ranked interacting pairs.
Saves figures to outputs/figures/ and tables to outputs/tables/, and dumps
the fitted model to outputs/models/ to show reproducibility.
"""
from __future__ import annotations
import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.ensemble import GradientBoostingClassifier

from _model_common import get_features, make_preprocessor, split_features

warnings.simplefilter("ignore")
rng = np.random.default_rng(123)

OUTCOME = "entry_from_zero_to_success_t1"
FSET = "main_t"
SEED = 123
GB_PARAMS = {"n_estimators": 150, "learning_rate": 0.03, "max_depth": 3}  # chosen params in importance table

df = pd.read_parquet(f"data/processed/model_sample_{OUTCOME}.parquet")
feats = get_features(FSET)
num, cat = split_features(feats)

tr = df[df["split_time"] == "train"].copy()
va = df[df["split_time"] == "valid"].copy()
te = df[df["split_time"] == "test"].copy()
trva = pd.concat([tr, va], ignore_index=True)

# Preprocessor: GB uses scale_numeric=False (median impute numeric, one-hot categoricals)
prep = make_preprocessor(feats, scale_numeric=False)
Xtrva = prep.fit_transform(trva[feats])
Xte = prep.transform(te[feats])
yte = te[OUTCOME].astype(int).to_numpy()
# Build transformed feature names manually (FunctionTransformer has no name passthrough).
# ColumnTransformer order: numeric block (in `num` order) then categorical one-hot block (in `cat` order).
ohe = prep.named_transformers_["cat"].named_steps["onehot"]
cat_names = []
for feat, cats in zip(cat, ohe.categories_):
    cat_names += [f"{feat}_{c}" for c in cats]
names = list(num) + cat_names
assert len(names) == Xtrva.shape[1], (len(names), Xtrva.shape[1])

gb = GradientBoostingClassifier(random_state=SEED, **GB_PARAMS)
gb.fit(Xtrva, trva[OUTCOME].astype(int))

os.makedirs("outputs/models", exist_ok=True)
joblib.dump({"preprocessor": prep, "model": gb, "feature_names": names},
            "outputs/models/gb_entry_from_zero_to_success_t1_main_t.joblib")

# ---- SHAP main effects on full test set ----
expl = shap.TreeExplainer(gb)
sv = expl.shap_values(Xte)  # (n_test, n_features) on margin/log-odds scale
if isinstance(sv, list):  # older API
    sv = sv[1]
sv = np.asarray(sv)
if sv.ndim == 3:
    sv = sv[:, :, 1]

def origin(colname: str) -> str:
    if colname.startswith("division_"):
        return "division (sector)"
    if colname.startswith("size_class_"):
        return "size_class"
    pretty = {
        "labor_productivity_vab": "labour productivity",
        "imports_comu": "EU import intensity",
        "imports_extra": "extra-EU import intensity",
        "import_experience": "import experience",
        "log_employment": "employment",
        "log_assets": "assets",
        "capital_intensity": "capital intensity",
        "wage_per_worker": "wage per worker",
        "equity_ratio": "equity ratio",
        "roa": "ROA",
        "profit_margin": "profit margin",
        "negative_equity": "negative equity",
    }
    return pretty.get(colname, colname)

mean_abs = np.abs(sv).mean(axis=0)
agg = {}
for c, m in zip(names, mean_abs):
    g = origin(c)
    agg[g] = agg.get(g, 0.0) + m
imp = pd.Series(agg).sort_values(ascending=False)
imp.to_csv("outputs/tables/shap_importance_entry_from_zero_to_success_t1_main_t.csv",
           header=["mean_abs_shap"])
print("=== SHAP global importance (mean|SHAP|, log-odds), original features ===")
print(imp.round(4).to_string())

# Bar figure
plt.figure(figsize=(8, 5))
imp.iloc[::-1].plot(kind="barh", color="#1f77b4")
plt.xlabel("Mean |SHAP| (impact on model log-odds)")
plt.title("SHAP feature importance: gradient boosting, time split")
plt.tight_layout()
plt.savefig("outputs/figures/fig_shap_importance.png", dpi=150)
plt.close()

# ---- SHAP dependence: labour productivity, colored by import_experience ----
prod_idx = names.index("labor_productivity_vab")
ie_idx = names.index("import_experience")
prod_val = Xte[:, prod_idx].astype(float)
prod_shap = sv[:, prod_idx]
ie = Xte[:, ie_idx].astype(int)
# clip productivity to robust range for display
lo, hi = np.nanpercentile(prod_val, [1, 99])
m = (prod_val >= lo) & (prod_val <= hi)
plt.figure(figsize=(8, 5))
for val, col, lab in [(0, "#1f77b4", "No import experience"), (1, "#ff7f0e", "Import experience")]:
    sel = m & (ie == val)
    plt.scatter(prod_val[sel], prod_shap[sel], s=6, alpha=0.3, color=col, label=lab)
plt.axhline(0, color="grey", lw=0.6)
plt.xlabel("Labour productivity (VAB per worker)")
plt.ylabel("SHAP value for labour productivity (log-odds)")
plt.title("SHAP dependence: productivity x import experience")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/fig_shap_dependence_productivity_import.png", dpi=150)
plt.close()

# ---- SHAP interaction values on a sample -> ranked pairs ----
nsamp = min(1500, Xte.shape[0])
idx = rng.choice(Xte.shape[0], size=nsamp, replace=False)
iv = expl.shap_interaction_values(Xte[idx])
if isinstance(iv, list):
    iv = iv[1]
iv = np.asarray(iv)
if iv.ndim == 4:
    iv = iv[:, :, :, 1]
# mean abs off-diagonal interaction, aggregated to original features
P = len(names)
groups = [origin(c) for c in names]
uniq = sorted(set(groups))
gidx = {g: [i for i, gg in enumerate(groups) if gg == g] for g in uniq}
mean_iv = np.abs(iv).mean(axis=0)  # (P,P)
pair_scores = {}
for a in range(len(uniq)):
    for b in range(a + 1, len(uniq)):
        ga, gb_ = uniq[a], uniq[b]
        block = mean_iv[np.ix_(gidx[ga], gidx[gb_])]
        # interaction counted twice (i,j)+(j,i); sum block and double
        pair_scores[(ga, gb_)] = 2.0 * block.sum()
pairs = pd.Series(pair_scores).sort_values(ascending=False)
pairs.index = [f"{a} x {b}" for (a, b) in pairs.index]
pairs.head(12).to_csv("outputs/tables/shap_interaction_pairs_entry_from_zero_to_success_t1_main_t.csv",
                      header=["mean_abs_shap_interaction"])
print("\n=== SHAP interaction strength (mean|interaction|, log-odds), top pairs ===")
print(pairs.head(12).round(4).to_string())

# productivity-specific interaction ranking
prod_group = "labour productivity"
prod_pairs = {k: v for k, v in pair_scores.items() if prod_group in k}
pp = pd.Series({f"{a} x {b}": v for (a, b), v in prod_pairs.items()}).sort_values(ascending=False)
print("\n=== productivity's strongest SHAP interactions ===")
print(pp.round(4).to_string())
print("\nDONE")
