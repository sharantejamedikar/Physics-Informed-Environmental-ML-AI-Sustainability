# UK Sewage Pollution Prediction & Deprivation Fairness Audit
**EEEM073 — AI and Sustainability | University of Surrey | 2025–2026**

---

## Overview

This project develops and evaluates a suite of machine learning models to predict annual sewage overflow frequency across England's storm overflow infrastructure, using data published by the Environment Agency. Beyond predictive performance, the project conducts a fairness audit to assess whether model accuracy differs systematically across communities of varying socioeconomic deprivation — addressing a gap in existing environmental AI literature.

The work is motivated by the ongoing UK sewage crisis: in 2024, water companies recorded 450,398 discharge events into rivers and coastal waters. This project asks whether AI can support more equitable environmental monitoring, or whether it risks perpetuating existing data inequalities.

This project contributes to **SDG 6** (Clean Water and Sanitation), **SDG 10** (Reduced Inequalities), **SDG 11** (Sustainable Cities and Communities), and **SDG 13** (Climate Action).

---

## Pipeline

Data Sources
├── EA EDM Annual Returns 2023–2024
└── English Indices of Multiple Deprivation (IMD) 2019
│
▼

Data Loading & Preprocessing
Feature engineering, spatial IMD join, temporal train/test split
│
▼
Exploratory Data Analysis
Spill distributions, company comparisons, deprivation analysis
│
▼
Modelling
XGBoost → LSTM → Attention-LSTM → PINN-LSTM
│
▼
Evaluation, Fairness Audit & XAI
Per-quartile performance metrics, SHAP attribution, policy recommendations
│
▼
Model Compression
Float16 quantisation, magnitude pruning (20/40/60%)

---

## Model Architecture

Four models are implemented in order of increasing complexity:

**XGBoost** serves as an interpretable tabular baseline, trained with documented hyperparameter tuning and early stopping. Its negative R² on the test set reveals that purely tabular approaches cannot capture site-specific temporal spillage dynamics, motivating the sequential models.

**LSTM with Site Embeddings** is a custom PyTorch implementation trained from scratch. Each of the 25,252 overflow sites receives a learned 16-dimensional embedding that encodes latent site behaviour beyond raw input features. Architecture: 2-layer LSTM, hidden dimension 64, dropout 0.3.

**Attention-LSTM** extends the LSTM with a self-attention mechanism over projected feature views, enabling instance-level attribution of which aspects of a site's profile drive each prediction. This provides interpretability beyond global SHAP values.

**PINN-LSTM** incorporates a physics residual loss term derived from a first-order sewage dilution equation:
dC/dt = −k · C + S(t)

The decay constant *k* is a learnable parameter. A lambda ablation study (λ ∈ {0.01, 0.1, 0.5}) determines the optimal weighting between data fit and physics constraint. The PINN achieves the best overall performance (RMSE = 32.05, R² = 0.291).

---

## Results Summary

| Model | RMSE | R² | Size (KB) |
|---|---|---|---|
| XGBoost | 48.86 | −0.649 | 2125.8 |
| LSTM | 35.51 | 0.129 | 1812.3 |
| Attention-LSTM | 45.15 | −0.408 | 1859.4 |
| **PINN-LSTM** | **32.05** | **0.291** | **1812.6** |

Float16 quantisation reduces model size by **49.9%** with no measurable RMSE degradation, enabling deployment on edge infrastructure without cloud dependency.

---

## Repository Structure
EEEM073/
├── data/
│   ├── raw/              ← source files (not tracked)
│   └── processed/        ← cleaned datasets and model outputs
├── notebooks/
│   ├── 1_data_loading_preprocessing.ipynb
│   ├── 2_eda.ipynb
│   ├── 3_modelling.ipynb
│   ├── 4_evaluation_fairness.ipynb
│   └── 5_compression.ipynb
├── outputs/              ← all figures (fig1–fig18)
├── README.md
└── requirements.txt

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/sharantejamedikar/EEEM073.git
cd EEEM073
```

**2. Create and activate the environment**
```bash
conda create -n eeem073 python=3.11 -y
conda activate eeem073
pip install -r requirements.txt
```

**3. Download the datasets**

| Dataset | Source | Destination |
|---|---|---|
| EDM Annual Returns 2023 & 2024 | [Environment Agency Open Data](https://environment.data.gov.uk/dataset/21e15f12-0df8-4bfc-b763-45226c16a8ac) | `data/raw/` |
| IMD 2019 File 7 | [MHCLG](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019) | `data/raw/imd_2019_file7.csv` |

**4. Run notebooks sequentially**

Each notebook saves outputs consumed by the next. Run in order 1 → 5. Estimated total runtime: 45–60 minutes on a standard CPU; faster with Apple Silicon MPS.

---

## Key Finding

Monitor reliability is **5.6 percentage points lower** in the most deprived quartile of communities compared to the least deprived. Sites with unreliable monitors record fewer spills — not because they spill less, but because events go unrecorded. The PINN-LSTM partially compensates for this by enforcing physically plausible predictions even in data-sparse areas, making it the most equitable model across deprivation groups.

---

## Author

**Sharan Teja Medikar**
MSc Artificial Intelligence, University of Surrey
EEEM073 — AI and Sustainability, 2025–2026