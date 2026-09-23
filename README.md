# Distance to Export — Python revision (Stage 1: diagnostics)

These four scripts replace `data_creationV2.R` and the data-prep half of
`predictions.R`. They do **not** train models. Their job is to make every
downstream design decision (outcome, horizon, size cells, split) for you,
from the data, with the leakage and definition fixes already applied.

> Do not write or run model code until script 04 has produced its counts.

## What was fixed relative to the old R pipeline
- **Leakage**: the old feature matrix kept `Total_Vendas` and `MercadoInterno_Vendas`,
  and the label was `1[(Total_Vendas − MercadoInterno_Vendas)/Total_Vendas ≥ 0.10]`.
  Script 02 measures this directly. These columns are kept here only to *build*
  outcomes/productivity and are never exposed as features.
- **Forward-looking design**: the outcome is future export entry among current
  non-exporters, so no contemporaneous sales variable can reconstruct the label.
- **Definitions matched to the paper**: productivity = `VAB / employment`
  (`VAB = Total_Vendas − CustoMercadoriasMateriasConsumidas_Total`),
  EU import share = `(EU Compras + EU Fornecimentos)/Total_Compras`,
  `capital_intensity = TotalActivo / employment` (not assets/sales),
  sector = 2-digit `Divisao` from `CAE.x`, manufacturing only.
- **Two outcome families, named differently**:
  - *true export entry* — candidate has zero exports at t (`entry_from_zero_*`);
  - *successful-export transition* — candidate is below 10% at t (`transition_below10_*`).

## Setup (Windows / VS Code)
```powershell
cd C:\Research\distance-to-export-python
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
Put `iesCompleto.RData` in `data\raw\`. Confirm the names in `config\config.yaml`
match what script 01 reports (fix there if your raw file differs).

## Run order
```powershell
python scripts/01_schema_check.py            # columns, year counts (2018 jump), missingness
python scripts/02_leakage_identity_check.py  # is the old label reconstructable? (expect yes)
python scripts/03_build_entry_panel.py       # -> data/processed/entry_panel.parquet
python scripts/04_design_diagnostics.py      # the gatekeeper: counts + recommendation
```

## What to read before going further
- `outputs/tables/leakage_identity_check.csv` — confirms the old AUCs were inflated.
- `outputs/tables/design_recommendation.csv` — which outcome/horizon is `headline candidate`
  vs `too sparse`, and the event counts **by size class** in the test window.
- `outputs/tables/pre_post_2018_composition.csv` — does 2018 change the size/exporter mix?
  If post-2018 is much more micro-heavy, the time split mixes a coverage change with the
  size-heterogeneity story; add the post-2018 robustness split.

Then write down, before touching models:
```
selected_outcome   = ...
selected_horizon   = ...
selected_size_cells = ...
selected_split     = ...   # and whether a post-2018 robustness split is needed
```

## One design choice the panel leaves open (deliberately)
The panel stores each predictor twice: contemporaneous at t (e.g. `labor_productivity_vab`)
and lagged one year (`L1_labor_productivity_vab`). The conservative design models on the
`L1_*` set (predictors strictly before the candidate year), which is safest but needs the
firm observed at t−1 and so costs sample. The standard forward design uses the at-t features
(predictors in the candidate year, outcome in the future). Decide after seeing how much
sample the `L1_` requirement removes — script 04's candidate counts use the outcome only,
so they are unaffected; the cost shows up later as missing `L1_` rows.

## Caveats
- I could not run these on the real confidential panel; treat the first run as a debugging
  pass and read the console output.
- Size classes here are employment-only bands, not the EU turnover+employment definition the
  paper used, so size counts won't match Table 1/A.3 exactly. Switch in `03` if you want exact
  comparability.
- Monetary levels are nominal. You have `CPI_2010-2021.xlsx`; decide whether to deflate the
  level features (ratios like ROA and import shares are scale-free and unaffected).

## Stage 2: modeling and referee-response tables

After scripts 01--04, the recommended baseline from your real diagnostics is:

```text
selected_outcome = entry_from_zero_to_success_t1
selected_horizon = 1
main test candidate years = 2019, 2020
main split = time split (firm_grouped_time is reported as a model-performance robustness check only)
main size cells = micro, small, medium+large
```

Before final modeling, rerun the patched scripts:

```powershell
python scripts/01b_duplicate_diagnostics.py
python scripts/03_build_entry_panel.py
python scripts/04_design_diagnostics.py
```

Then run Stage 2:

```powershell
python scripts/05_make_splits.py --outcome entry_from_zero_to_success_t1 --include-robustness-splits
python scripts/06_run_policy_baselines.py --outcome entry_from_zero_to_success_t1 --split-designs time
python scripts/07_run_main_models.py --outcome entry_from_zero_to_success_t1 --feature-set main_t --split-designs time --quick
python scripts/08_evaluate_and_bootstrap.py --outcome entry_from_zero_to_success_t1 --feature-set main_t --bootstrap 300
python scripts/09_size_and_mr_comparison.py --outcome entry_from_zero_to_success_t1 --feature-set main_t --split-design time
python scripts/10_interactions.py --outcome entry_from_zero_to_success_t1 --feature-set main_t --split-design time --model gradient_boosting
```

Remove `--quick` once the pipeline runs cleanly.

For conservative robustness:

```powershell
python scripts/07_run_main_models.py --outcome entry_from_zero_to_success_t1 --feature-set main_l1 --split-designs time --quick
python scripts/08_evaluate_and_bootstrap.py --outcome entry_from_zero_to_success_t1 --feature-set all --bootstrap 300
```

The main outputs for the paper are:

```text
outputs/tables/split_diagnostics_entry_from_zero_to_success_t1.csv
outputs/tables/policy_baseline_targeting_entry_from_zero_to_success_t1.csv
outputs/tables/main_model_metrics_entry_from_zero_to_success_t1_main_t.csv
outputs/tables/model_performance_with_ci_entry_from_zero_to_success_t1.csv
outputs/tables/size_group_prediction_performance_entry_from_zero_to_success_t1.csv
outputs/tables/size_group_retrained_performance_entry_from_zero_to_success_t1.csv
outputs/tables/permutation_importance_entry_from_zero_to_success_t1_main_t.csv
outputs/tables/interaction_strength_entry_from_zero_to_success_t1_main_t.csv
```
