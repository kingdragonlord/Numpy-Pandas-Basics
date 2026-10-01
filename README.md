# NumPy & Pandas Basics: Core Operations & Titanic Survival Classification

A hands-on notebook demonstrating foundational numerical operations in NumPy and an end-to-end exploratory data analysis (EDA), preprocessing, and classification pipeline using Pandas and Scikit-Learn on the Titanic dataset.

---

## Overview

This notebook is divided into two primary sections:
1. **NumPy Array Manipulations & Vectorization Exercises**: Implementing custom vector operations, broadcasting techniques, distance metrics, and row-wise filtering.
2. **Titanic Survival Prediction Pipeline**: Data ingestion, exploratory analysis, handling missing values, categorical encoding, feature scaling using the custom standardization function, and K-Nearest Neighbors (KNN) classification.

---

## Notebook Structure & Key Implementations

### Part 1: NumPy Exercises

| # | Task | Method / Implementation Details |
|---|---|---|
| **1** | **Z-Score Normalization** | Implemented `z_score_normalization()` to calculate column-wise standardization using `(X - mean) / std` without in-place mutation. Supports optional returns for column means and standard deviations. |
| **2** | **Closest Value to Scalar** | Implemented both iterative and vectorized methods (`np.argmin(np.abs(n - scalar), axis=0)`) to extract the closest scalar match per column. |
| **3** | **Pairwise Euclidean Distances** | Computed full pairwise distance matrices using both nested loops with `np.linalg.norm` and vectorized 3D array broadcasting (`A[:, np.newaxis] - A[np.newaxis, :]`). |
| **4** | **Sum of Absolute Values** | Implemented axis-aware reductions using `np.abs(A).sum(axis=axis)` across rows (`axis=1`) and columns (`axis=0`). |
| **5** | **Second Smallest Element** | Extracted the second smallest element across rows using both row-wise sorting (`np.sort(A, axis=1)[:, 1]`) and list-based iteration. |
| **6** | **Non-Unique Row Filtering** | Generated a `(30, 20)` integer matrix and extracted all duplicate elements per row using `np.unique(..., return_counts=True)` combined with boolean masking via `np.isin()`. |

---

### Part 2: Titanic Survival Classification (Pandas & Scikit-Learn)

#### 1. Data Cleaning & Preprocessing
* **Missing Value Imputation**:
  * Imputed missing `Age` values using the median (~28.0 years).
  * Dropped the high-cardinality/sparse `Cabin` column.
  * Imputed missing `Embarked` entries using the mode (`'S'`).
* **Categorical Encoding**:
  * Applied `pd.get_dummies(..., drop_first=True)` to convert `Sex` and `Embarked` into numerical indicator variables (`Sex_male`, `Embarked_Q`, `Embarked_S`).
* **Selected Feature Set**:
  * `Pclass`, `Sex_male`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked_Q`, `Embarked_S`

#### 2. Train/Test Split & Standardization
* **Split**: 80% training set, 20% test set (`random_state=42`).
* **Data Leakage Prevention**:
  * Standardized `X_train` using the custom `z_score_normalization()` function.
  * Standardized `X_test` using the learned training distribution statistics ($\mu_{train}$ and $\sigma_{train}$).

#### 3. Model Training & Evaluation
* **Model**: `KNeighborsClassifier(n_neighbors=5)`
* **Evaluation**: Model accuracy score and confusion matrix inspection via Scikit-Learn.

---

## Results & Model Performance

* **Classification Algorithm**: K-Nearest Neighbors ($k = 5$)
* **Test Set Accuracy**: **80.45%** (`0.8044692737430168`)
* **Key Takeaways**:
  * Vectorized array broadcasting in NumPy matches manual iterative approaches while significantly reducing computational overhead.
  * Normalizing feature columns via training set statistics prevents data leakage and enables distance-based classifiers (KNN) to balance features of varying scales (e.g., `Fare` vs. `Age` vs. binary indicators).

---

## Requirements

To run this notebook locally or in Google Colab, install the required libraries:

```bash
pip install numpy pandas scikit-learn matplotlib
