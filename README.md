# Predicting Successful Export Entry: A Machine Learning Approach with Portuguese Firms

Replication code for Barbosa, P. H., Amador, J. and Cortes, J. (2026), *Predicting Successful Export Entry: A Machine Learning Approach with Portuguese Firms*.

The code builds a forward-looking firm-year panel of Portuguese manufacturing firms, defines the export-entry outcomes, creates the validation splits, estimates and evaluates the models, and writes the tables and figures reported in the paper.

## Data availability

The analysis uses confidential firm-level data from *Informação Empresarial Simplificada* (IES) for 2010–2021. The microdata cannot be redistributed. This repository contains only source code, configuration templates, dependency specifications and execution instructions. It contains no raw or processed microdata, firm identifiers, firm-level predictions or fitted models derived from confidential data.

The analysis was conducted outside the Banco de Portugal Microdata Research Laboratory (BPLIM). Authorised researchers may reconstruct the workflow with a comparable Central Balance Sheet/IES extract from BPLIM, subject to project approval and the applicable confidentiality and output-control procedures. If an IES item required here is not in the standard extract, it can be requested through BPLIM by identifying the corresponding IES table and item. See Appendix D of the paper and the [BPLIM access page](https://bplim.bportugal.pt/content/access-0).

## Requirements

- Python 3.13.6
- The packages in `requirements.txt`, including scikit-learn 1.8.0 and shap 0.52.0

```powershell
python -m venv .venv
.\.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Place the raw IES extract in `data/raw/` and adapt `config/config.yaml` to the variable names in your extract. The original analysis used an R data file (`iesCompleto.RData`). Script 01 lists the columns found in the raw file, so the mapping can be checked before the panel is built. The `data/` and `outputs/` folders are local and are not part of this repository.

## Design

- **Candidates.** Manufacturing firm-years with zero foreign goods sales in year *t* and an observed outcome in *t*+1.
- **Headline outcome** (`entry_from_zero_to_success_t1`). Foreign goods sales reach at least 10% of total goods sales in *t*+1. Outcome columns come in two families: `entry_from_zero_*` for candidates with zero exports at *t*, and `transition_below10_*` for candidates below 10% export intensity at *t*.
- **Predictors.** Measured at *t* (feature set `main_t`). The panel also stores one-year lags (`L1_*`), which the paper does not use. Size classes are employment-based (micro < 10, small 10–49, medium 50–249, large ≥ 250 employees). Sector is the two-digit NACE Rev.2 division.
- **No sales decomposition in the features.** The contemporaneous exporter label used in an earlier version of the paper is exactly reconstructable from total and domestic sales (script 02; Table A.5 of the paper). Total and domestic sales are therefore used only to build the outcomes and derived ratios such as value-added productivity, and never enter the feature matrix directly.
- **Validation.** Main chronological split: training on candidate years 2010–2017, validation on 2018, test on 2019–2020 (outcomes observed in 2020–2021). Hyperparameters are selected by PR-AUC on the validation window, and the selected specification is refitted on training plus validation. A single random seed (123) governs the splits, model fitting and the firm-clustered bootstrap.

## Run order

The commands below reproduce the main results for the chronological split.

```powershell
python scripts/01_schema_check.py                # raw-extract columns, year counts, missingness
python scripts/01b_duplicate_diagnostics.py      # duplicate firm-year records
python scripts/02_leakage_identity_check.py      # sales identity check (Table A.5)
python scripts/03_build_entry_panel.py           # -> data/processed/entry_panel.parquet
python scripts/04_design_diagnostics.py          # event counts by outcome, horizon and size class
python scripts/05_make_splits.py --outcome entry_from_zero_to_success_t1 --include-robustness-splits
python scripts/06_run_policy_baselines.py --outcome entry_from_zero_to_success_t1 --split-designs time
python scripts/07_run_main_models.py --outcome entry_from_zero_to_success_t1 --feature-set main_t --split-designs time
python scripts/08_evaluate_and_bootstrap.py --outcome entry_from_zero_to_success_t1 --feature-set main_t --bootstrap 1000
python scripts/09_size_and_mr_comparison.py --outcome entry_from_zero_to_success_t1 --feature-set main_t --split-design time
python scripts/10_interactions.py --outcome entry_from_zero_to_success_t1 --feature-set main_t --split-design time --model gradient_boosting
```

## Main outputs

Tables are written to `outputs/tables/`.

| Output file | Paper exhibit |
|---|---|
| `leakage_identity_check.csv` | Table A.5 |
| `split_diagnostics_entry_from_zero_to_success_t1.csv` | Table A.1 |
| `main_model_metrics_entry_from_zero_to_success_t1_main_t.csv` | Table 3 |
| `model_performance_with_ci_entry_from_zero_to_success_t1.csv` | Table A.2 |
| `policy_baseline_targeting_entry_from_zero_to_success_t1.csv` | Table 8 (simple targeting rules) |
| `size_group_retrained_performance_entry_from_zero_to_success_t1.csv` | Tables 7 and A.3 |
| `permutation_importance_entry_from_zero_to_success_t1_main_t.csv` | Table B.1 |
| `interaction_strength_entry_from_zero_to_success_t1_main_t.csv` | Table B.2 |

## Contact

Paulo Henrique Barbosa (paulofbarbosa5@gmail.com)
