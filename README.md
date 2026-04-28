# 🌊 UK Sewage Pollution & Deprivation Fairness Audit
### EEEM073 — AI and Sustainability | University of Surrey | 2025–2026

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)
![University](https://img.shields.io/badge/University-Surrey-1B4F8A)

> *Can AI-driven sewage spill prediction fairly serve all communities in England or does it perpetuate existing environmental inequalities?*

---

## 📌 Project Overview

Water companies in England discharged sewage **450,398 times in 2024**.
Yet monitoring reliability varies significantly by area deprivation — meaning
the true pollution burden on vulnerable communities may be systematically
undercounted.

This project builds, audits and compresses three AI models to predict annual
sewage overflow frequency across 14,000+ sites across England, then conducts
a fairness audit across IMD deprivation quartiles to assess whether model
performance is equitable across communities.

---

## 🔬 Key Findings

| Finding | Result |
|---|---|
| Best model | **PINN-LSTM** (RMSE=32.05, R²=0.291) |
| Physics constraint benefit | PINN outperforms plain LSTM by **9.7% RMSE** |
| Monitor reliability gap | **5.6 percentage points** lower in most vs least deprived areas |
| Dominant predictor (SHAP) | Spill severity score (frequency × duration) |
| Float16 size reduction | **49.9%** with **0% RMSE degradation** |
| Core fairness finding | Deprived areas have less reliable monitors → true spill burden undercounted |

---

## 🏗️ Pipeline
Raw Data
├── EA EDM Annual Returns 2023–2024  (14,000+ overflow sites × 2 years)
└── English IMD 2019                 (deprivation scores × 33k LSOAs)
│
▼
┌──────────────────────────────┐
│  1. Data Loading &           │──► edm_processed.csv (28,688 rows, 27 features)
│     Preprocessing            │    edm_site_imd.csv  (28,632 rows, 34 features)
└──────────────────────────────┘
│
▼
┌──────────────────────────────┐
│  2. Exploratory Data         │──► 7 figures — spill distributions,
│     Analysis                 │    company performance, deprivation analysis,
│                              │    undercounting hypothesis
└──────────────────────────────┘
│
▼
┌──────────────────────────────┐
│  3. Modelling                │──► XGBoost baseline
│                              │    Custom LSTM with site embeddings
│                              │    PINN-LSTM (sewage dilution physics)
│                              │    Lambda ablation: λ ∈ {0.01, 0.1, 0.5}
└──────────────────────────────┘
│
▼
┌──────────────────────────────┐
│  4. Evaluation, Fairness     │──► Fairness audit by IMD quartile
│     Audit & XAI              │    SHAP (XGBoost + LSTM GradientExplainer)
│                              │    Policy lever table
└──────────────────────────────┘
│
▼
┌──────────────────────────────┐
│  5. Model Compression        │──► Float16 quantisation (−49.9% size, 0% RMSE loss)
│                              │    Magnitude pruning (20%, 40%, 60%)
│                              │    Fairness gap measured pre/post compression
└──────────────────────────────┘
---

## 📊 Model Performance (2024 Test Set)

| Model | RMSE | MAE | R² | NSE | Size (KB) | Inf Time (s) |
|---|---|---|---|---|---|---|
| XGBoost (baseline) | 48.860 | 31.259 | -0.649 | -0.649 | 2125.8 | 0.006 |
| LSTM | 35.513 | 19.000 | 0.129 | 0.129 | 1812.3 | 0.086 |
| **PINN-LSTM** | **32.053** | **17.395** | **0.291** | **0.291** | **1812.6** | **0.084** |

> NSE = Nash-Sutcliffe Efficiency — standard metric in environmental modelling.
> NSE=1 is perfect, NSE=0 equals mean prediction, NSE<0 is worse than mean.

---

## 🗜️ Compression Results

| Model | Compression | RMSE | Size (KB) | Size Reduction | RMSE Change |
|---|---|---|---|---|---|
| LSTM | Baseline (Float32) | 35.513 | 1812.3 | — | — |
| LSTM | **Float16** | **35.511** | **906.2** | **−49.9%** | **+0.000** |
| LSTM | Int8 simulated | 35.513 | 1812.3 | ~0% | +0.000 |
| LSTM | Pruned 20% | 34.966 | 1812.3 | ~0% | −0.547 |
| LSTM | Pruned 40% | 35.128 | 1812.3 | ~0% | −0.385 |
| LSTM | Pruned 60% | 42.800 | 1812.3 | ~0% | +7.287 |
| PINN-LSTM | Baseline (Float32) | 32.053 | 1812.6 | — | — |
| PINN-LSTM | **Float16** | **32.053** | **906.3** | **−49.9%** | **+0.000** |
| PINN-LSTM | Int8 simulated | 32.053 | 1812.6 | ~0% | +0.000 |
| PINN-LSTM | Pruned 20% | 31.900 | 1812.6 | ~0% | −0.153 |
| PINN-LSTM | Pruned 40% | 34.200 | 1812.6 | ~0% | +2.147 |
| PINN-LSTM | Pruned 60% | 41.500 | 1812.6 | ~0% | +9.447 |

> **Best compression strategy: Float16** — halves model size with zero measurable
> accuracy loss. Enables deployment on EA edge hardware without cloud dependency.

---

## 🗂️ Repository Structure
EEEM073/
├── data/
│   ├── raw/                          ← Source files (not tracked in git)
│   │   ├── EDM_2023_Storm_Overflow_Annual_Return/
│   │   ├── EDM_2024_Storm_Overflow_Annual_Return/
│   │   └── imd_2019_file7.csv
│   └── processed/                    ← Cleaned datasets & model outputs
│       ├── edm_processed.csv
│       ├── edm_site_imd.csv
│       ├── model_comparison.csv
│       ├── fairness_metrics.csv
│       ├── compression_results.csv
│       └── policy_lever_table.csv
├── notebooks/
│   ├── 1_data_loading_preprocessing.ipynb
│   ├── 2_eda.ipynb
│   ├── 3_modelling.ipynb
│   ├── 4_evaluation_fairness.ipynb
│   └── 5_compression.ipynb
├── outputs/                          ← All 18 figures
├── README.md
└── requirements.txt

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/sharantejamedikar/EEEM073.git
cd EEEM073
```

### 2. Set up environment
```bash
conda create -n eeem073 python=3.11 -y
conda activate eeem073
pip install -r requirements.txt
```

### 3. Download datasets
| Dataset | Source | Save to |
|---|---|---|
| EDM Annual Returns 2023 | [EA Open Data](https://environment.data.gov.uk/dataset/21e15f12-0df8-4bfc-b763-45226c16a8ac) | `data/raw/EDM_2023_Storm_Overflow_Annual_Return/` |
| EDM Annual Returns 2024 | [EA Open Data](https://environment.data.gov.uk/dataset/21e15f12-0df8-4bfc-b763-45226c16a8ac) | `data/raw/EDM_2024_Storm_Overflow_Annual_Return/` |
| IMD 2019 File 7 | [Gov.uk](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019) | `data/raw/imd_2019_file7.csv` |

### 4. Run notebooks sequentially
Each notebook saves outputs consumed by the next — run in order:
1_data_loading_preprocessing.ipynb  →  produces edm_processed.csv
2_eda.ipynb                         →  produces 7 figures
3_modelling.ipynb                   →  produces lstm_best.pt, pinn_best.pt
4_evaluation_fairness.ipynb         →  produces fairness_metrics.csv
5_compression.ipynb                 →  produces compression_results.csv

### 5. Hardware requirements
- Tested on MacBook Pro M4 Pro (Apple Silicon)
- PyTorch MPS acceleration auto-detected
- No GPU required — all notebooks run on CPU if MPS unavailable
- Estimated total runtime: ~45 minutes

---

## 🤖 Model Architecture

### 1. XGBoost Baseline
Gradient boosted tree model. Fast, interpretable, SHAP-compatible.
Negative R² on test set reveals purely tabular approach cannot capture
site-specific temporal spillage dynamics — motivating the LSTM.

**Key hyperparameters:** `n_estimators=500`, `max_depth=6`,
`learning_rate=0.05`, `subsample=0.8`, early stopping at 30 rounds.

### 2. Custom LSTM with Site Embeddings
Built from scratch in PyTorch. Each of the 25,252 overflow sites
receives a learned 16-dimensional embedding vector capturing
site-specific behaviour beyond raw features.
Input (9 features) + Site Embedding (16 dims)
→ LSTM (2 layers, hidden=64, dropout=0.3)
→ FC(64→32) → ReLU → FC(32→1)
→ Predicted spill count

**Design choices documented:** sequence length, hidden dimensions,
dropout rate, embedding size — all tuned with validation curves.

### 3. PINN-LSTM (Physics-Informed)
Extends the LSTM with a physics residual loss enforcing
sewage dilution dynamics: 
dC/dt = −k·C + S(t)
Where C = pollutant concentration (proxied by spill count),
k = **learnable** decay constant, S = source discharge rate.

**Combined loss:**
L_total = L_data + λ · L_physics

Lambda ablation study (λ ∈ {0.01, 0.1, 0.5}) determines
optimal physics/data weighting. The physics constraint
prevents physically implausible predictions — particularly
important for data-sparse deprived area sites.

---

## ⚖️ Fairness Audit

A key contribution of this project is auditing whether model performance
differs systematically across IMD deprivation quartiles.

**Finding:** Monitor reliability is **5.6 percentage points lower**
in the most deprived quartile vs the least deprived. Sites with
unreliable monitors record fewer spills — not because they spill less,
but because events go unrecorded. The PINN-LSTM partially compensates
for this by enforcing physical plausibility in sparse-data regions.

**Fairness metrics computed:** RMSE, MAE, Mean Prediction Error
per deprivation quartile for all three models.

---

## 🌍 SDG Alignment

| SDG | Connection |
|---|---|
| **SDG 6** — Clean Water & Sanitation | Improving detection of sewage pollution events |
| **SDG 10** — Reduced Inequalities | Fairness audit ensures AI doesn't disadvantage deprived communities |
| **SDG 11** — Sustainable Cities | Supporting urban water infrastructure monitoring |
| **SDG 13** — Climate Action | Sewage overflow risk increases with extreme rainfall events |

---

## 📦 Dependencies
torch>=2.1.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
shap>=0.43.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
openpyxl>=3.1.0
---

## 📄 Data Sources

- [EA Event Duration Monitoring — Storm Overflow Annual Returns](https://environment.data.gov.uk/dataset/21e15f12-0df8-4bfc-b763-45226c16a8ac)
- [English Indices of Multiple Deprivation 2019 — MHCLG](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019)

---

## 👤 Author

**Sharan Teja Medikar**
MSc Artificial Intelligence, University of Surrey
Module: EEEM073 — AI and Sustainability (2025–2026)