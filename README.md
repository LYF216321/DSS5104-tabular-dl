# DSS5104 — Deep Learning for Tabular Data

## Overview
This project evaluates deep learning methods (TabNet, FT-Transformer) against 
classical baselines (XGBoost, Logistic/Ridge Regression) on three tabular 
datasets. All experiments follow a rigorous protocol with fixed data splits, 
3 random seeds, and Optuna hyperparameter tuning.

## Repository Structure
├── notebook_1_baselines.ipynb        # XGBoost + Logistic/Ridge Regression
├── notebook_2_tabnet.ipynb           # TabNet + Optuna tuning
├── notebook_3_fttransformer.ipynb    # FT-Transformer + Optuna tuning
├── notebook_4_analysis.ipynb         # Results aggregation + figures
├── requirements.txt                  # Python dependencies
└── README.md

## Datasets
| Dataset | Task | Rows used |
|---------|------|-----------|
| California Housing | Regression | 20,640 (full) |
| Adult Income | Binary Classification | 48,842 (full) |
| Covertype | Multi-class Classification | 20,000 (subsampled) |

All datasets are loaded programmatically via scikit-learn or UCI repository.
No manual download required.

## Experimental Protocol
- **Data split**: 60% train / 20% validation / 20% test (fixed)
- **Random seeds**: 0, 42, 123 (results reported as mean ± std)
- **Hyperparameter tuning**: Optuna, 20 trials per model per dataset
- **Metrics**:
  - Regression: RMSE, MAE, R²
  - Classification: Accuracy, AUC-ROC, F1-score

## How to Reproduce
### Option 1: Kaggle (recommended for baselines)
1. Open `notebook_1_baselines.ipynb` in Kaggle
2. Enable GPU T4
3. Run all cells

### Option 2: Google Colab (recommended for deep learning models)
1. Open `notebook_2_tabnet.ipynb` or `notebook_3_fttransformer.ipynb` in Colab
2. Enable GPU T4
3. Mount Google Drive and update `SAVE_DIR` path
4. Run cells sequentially

### Install dependencies
```bash
pip install -r requirements.txt
```

## Results Summary
| Dataset | Linear | XGBoost | TabNet | FT-Transformer |
|---------|--------|---------|--------|----------------|
| California (R²) | 0.598 | **0.846** | 0.678 | 0.825 |
| Adult (Accuracy) | 0.825 | **0.876** | 0.854 | 0.844 |
| Covertype (Accuracy) | 0.716 | 0.848 | 0.819 | **0.859** |

## Key Findings
- XGBoost achieves the best overall performance on small-to-medium datasets
- FT-Transformer outperforms XGBoost on Covertype, suggesting deep models 
  are competitive when datasets have many features
- Linear models and XGBoost are significantly more stable across random seeds
- Deep learning models incur substantially higher training and inference costs

## References
See report PDF for full references.
