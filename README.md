# Physics-Informed Environmental ML for Sewage Spill Analysis

**EEEM073 — AI and Sustainability | University of Surrey (2025–2026)**

This repository contains an environmental machine-learning project completed for EEEM073, comparing **XGBoost**, **LSTM**, and a **physics-informed LSTM** on UK sewage-overflow data.

## Project Scope

- **Dataset scale:** 28,632 site-year observations across 26,446 locations
- **Task framing:** annual spill-count modelling using engineered operational and contextual features
- **Experimental design:** temporal split (train on 2023, evaluate on 2024) for model comparison
- **Analysis layers:** predictive performance, interpretability, and environmental-equity analysis across deprivation groups

## Modelling Approach

The workflow builds and compares:

1. XGBoost baseline
2. LSTM with site-level sequential representation
3. Physics-informed LSTM with a residual physics constraint

The physics-informed variant is evaluated as part of this comparative experiment, alongside feature-attribution and subgroup error analysis.

## Compression Result

Post-training float16 compression reduced model size from approximately **1.8 MB** to approximately **908 KB** in repository outputs.

## Technical Limitation

The original experiment used several engineered features derived from spill-count outcomes, so the reported model comparison is treated as **exploratory modelling** rather than leakage-free prospective forecasting. A next iteration should rebuild features using only information available before the prediction period and then rerun the baseline and physics-informed comparison under that stricter setup.

## Repository Structure

- `notebooks/` — end-to-end workflow (data prep, EDA, modelling, evaluation, compression)
- `outputs/` — generated figures and artefacts
- `data/processed/` — processed datasets and model outputs
- `requirements.txt` — Python dependencies

## Author

Sharan Teja Medikar
MSc Artificial Intelligence, University of Surrey
