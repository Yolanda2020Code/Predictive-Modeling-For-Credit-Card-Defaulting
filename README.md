# Credit Card Default Prediction

MSc Artificial Intelligence — Practical Assignment
Author: Yolanda N. Nkala · Supervisor: Dr. Abdelaziz Triki · Due: 10 June 2026

## Overview

An end-to-end supervised machine learning pipeline that predicts whether a credit card holder will default on their next payment. The project works through a binary classification problem on a heavily imbalanced dataset, prioritising the correct identification of defaulters (the minority class) over raw accuracy. The full workflow runs in a single Jupyter notebook covering exploratory analysis, leakage-safe data preparation, training of six models, evaluation, hyperparameter tuning, and interpretability.

## Dataset

- **File:** `Credit_Card.csv`
- **Records:** 34,788
- **Features:** 30 (demographic, credit-limit, bill-amount, payment-amount, and repayment-status fields, plus a high-cardinality `CITY` field)
- **Target:** binary — default (1) vs. no default (0), with significant class imbalance

The dataset is loaded via `io.BytesIO` to avoid partial-row streaming issues in hosted notebook environments.

## Notebook structure

The notebook (`Yolanda_N_Nkala_MSc_AI_Practical_Assignment_Dr_Abdelaziz_Triki_10_06_2026.ipynb`) is organised into the following tasks:

- **Setup** — library imports (all consolidated in one cell), dataset loading, and feature-type definitions.
- **Task 1 — Exploratory Data Analysis:** target distribution, missing-value audit, descriptive statistics, univariate and bivariate analysis, outlier detection, and correlation analysis.
- **Task 2 — Data Preparation:** train/test split first, then all transformers fit on the training set only. Median imputation, Winsorisation at the 1st/99th percentiles, one-hot encoding for nominal features, target encoding for the high-cardinality `CITY` field, and standard scaling. Repayment-status (`PAY_*`) columns are retained as ordinal integers, and known leakage columns are dropped.
- **Task 3 — Model Training:** six classifiers (Logistic Regression, Decision Tree, Random Forest, SVM, KNN, and Gradient Boosting), with class imbalance handled via `class_weight="balanced"` where supported and `sample_weight` for Gradient Boosting.
- **Task 4 — Evaluation & Visualisation:** classification reports, confusion matrices, ROC and Precision-Recall curves, a model comparison table sorted by recall, hyperparameter tuning with `GridSearchCV` (scored on `roc_auc`), and SHAP-based interpretability of the tuned model.
- **Task 5 — Conclusion:** summary of findings, limitations, and future work.

## Methodology highlights

- **Leakage-safe design:** the data is split before any fitting, and every transformer learns its parameters from the training set only. Raw partitions (`X_train_raw` / `X_test_raw`) are preserved before transformation for use in the end-to-end pipeline.
- **Recall-first evaluation:** because missing a defaulter is costlier than a false alarm, models are ranked primarily by recall on the positive class, supported by F1 and PR-AUC.
- **Reproducible pipelines:** a `ColumnTransformer` plus per-model `Pipeline` objects (using `clone`) keep preprocessing and estimation encapsulated and repeatable, with a fixed `random_state=42`.

## Requirements

- Python 3.x
- numpy, pandas, matplotlib, seaborn, scipy
- scikit-learn
- category_encoders
- shap

Install dependencies with:

```
pip install numpy pandas matplotlib seaborn scipy scikit-learn category_encoders shap
```

(`category_encoders` is also installed inline at the top of the notebook.)

## How to run

1. Place `Credit_Card.csv` alongside the notebook (or update the load path).
2. Open the notebook and run the cells in order. The first code cell installs and imports all dependencies; later cells depend on objects created earlier, so sequential execution is required.
3. Generated figures are saved to disk as `.png` files (for example, the scaling demo, performance curves, recall comparison, and SHAP summary plot).

## Deliverables

- Jupyter notebook — full pipeline (Tasks 1–5).
- Word report — written analysis with inline Harvard citations.

## References

Key sources cited in the report include He and Garcia (2009), Wilcox (2022), Bergstra and Bengio (2012), Géron (2022), Hosmer et al. (2013), Cortes and Vapnik (1995), Breiman (2001), and Friedman (2001). Full details appear in the References section of the notebook and report.
