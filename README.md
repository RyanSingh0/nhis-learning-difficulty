# Learning Difficulty Screening — NHIS 2024

> **MET CS 699 · Boston University**
> Team: Aryan Meena · Aman Nishad

![R](https://img.shields.io/badge/R-4.3-blue)
![Models](https://img.shields.io/badge/Models%20Trained-36-orange)
![Algorithms](https://img.shields.io/badge/Algorithms-9-brightgreen)
![Best](https://img.shields.io/badge/Best%20Model-XGBoost-green)

---

## Overview

End-to-end classification pipeline to **screen children for learning difficulties** using the 2024 National Health Interview Survey (NHIS). 36 models trained across 9 algorithms and 4 balancing strategies, with deterministic cross-validation and rigorous evaluation.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | 2024 National Health Interview Survey (NHIS) — Sample Child File |
| Records | **7,439 children** aged 0–17 years |
| Features | **353 variables** (demographics, health, socioeconomic, behavioral) |
| Target | `Class`: Yes (learning difficulty) / No |
| Class distribution | Imbalanced — minority class: children with learning difficulty |

**Feature domains:** demographics, household structure, socioeconomic indicators, physical health/chronic conditions, neurodevelopmental diagnoses (ADHD, autism, developmental delay), behavioral indicators.

---

## Preprocessing Pipeline

```
Raw NHIS data (7,439 × 353)
   ↓ Missing code normalization (empty strings, −9, −99, 7/8/9, 97/98/99 → NA)
   ↓ High-missingness drop (≥90% missing → removed)
   ↓ Near-zero-variance removal (caret::nearZeroVar)
   ↓ Median imputation (numerics) + Mode imputation (categoricals)
   ↓ Winsorization (1st/99th percentile — stabilizes outlier effects)
   ↓ ID/near-identifier removal (>90% unique cardinality)
   ↓ Correlation pruning (r ≥ 0.80 — reduces multicollinearity)
   ↓ Stratified 80/20 split
   ↓ Rare factor level consolidation (<10 obs → "Other")
   ↓ Center/scale (training statistics only — no leakage)
   ↓ 4 balanced training sets created
```

---

## 36 Models: 9 Algorithms × 4 Balancing Strategies

**Algorithms:**
| # | Algorithm | Type |
|---|-----------|------|
| 1 | Logistic Regression | Linear |
| 2 | Decision Tree | Non-linear |
| 3 | Naïve Bayes | Probabilistic |
| 4 | k-NN | Instance-based |
| 5 | SVM (RBF kernel) | Kernel method |
| 6 | MLP (Neural Network) | Deep |
| 7 | Random Forest | Ensemble |
| 8 | **XGBoost** | **Gradient boosting** |
| 9 | Bagging | Ensemble |

**Balancing strategies:**
| Strategy | Method | Advantage |
|----------|--------|-----------|
| Downsample | Undersample majority class | Fast, prevents majority dominance |
| Upsample | Oversample minority class | Retains all data |
| SMOTE | Synthetic minority oversampling | Creates realistic synthetic samples |
| NearMiss-1 | Intelligent undersampling | Removes borderline majority samples |

**Reproducibility:** Fixed random seeds, frozen CV folds (3-fold), saved model artifacts. All 36 builds are deterministic.

---

## Results

**Best model: XGBoost + Downsampling**

| Metric | Value |
|--------|-------|
| Selection criterion | Minimum TPR constraints + composite score (ROC + class-wise TPR) |
| Best model | **XGBoost + downsampling** — exceeds all minimum TPR requirements |

Model selection uses a composite score combining minimum True Positive Rate (TPR) constraints with ROC-AUC and class-wise TPRs. The goal is a clinically useful screening tool: it must catch a minimum fraction of children with learning difficulties, not just maximize overall accuracy.

---

## Key Design Choices

**Why freeze CV folds?** Random CV splits mean different runs compare different data subsets — not the same folds. Frozen folds make all 36 models comparable on identical data partitions.

**Why composite scoring vs simple AUC?** A screening tool that never predicts "Yes" could have decent AUC but zero utility. The composite score enforces a minimum recall floor before considering other metrics.

**Why all 4 balancing strategies?** The best strategy varies by algorithm. SMOTE that helps LR may hurt kNN. Running all 4 finds the optimal pairing per algorithm.

---

## How to Run

```r
# Install dependencies
install.packages(c("caret", "xgboost", "ROSE", "DMwR2", "ggplot2", "dplyr"))

# Run pipeline
source("src/classification_pipeline.R")
```

---

**Aryan Meena** · Aman Nishad
Boston University, MET CS 699
