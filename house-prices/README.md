# House Prices: Advanced Regression Techniques

End-to-end tabular regression pipeline predicting residential home sale prices in Ames, Iowa (`RMSLE` evaluation metric), built with leak-free `scikit-learn` pipelines, domain feature engineering, Optuna Bayesian hyperparameter optimization, and out-of-fold model blending.

## Key Engineering Highlights
- **Target Transformation & Outlier Handling:** Applied `np.log1p(SalePrice)` to reduce target skewness from `1.88` to `0.12` and align training loss directly with RMSLE; removed 2 documented partial-sale outliers (`GrLivArea > 4500`, `SalePrice < $300k`).
- **Leak-Free Feature Engineering (`src/features.py`):** Custom `BaseEstimator`/`TransformerMixin` distinguishing structural `"None"`/`0` absences (pools, basements, garages) from true missing values, imputing `LotFrontage` via training-fold `Neighborhood` medians, mapping ordinal quality ratings (`Ex`–`Po` $\rightarrow$ `5`–`0`), and engineering `TotalSF`, `TotalBath`, `TotalPorchSF`, `HouseAge`, and `RemodAge`.
- **Bayesian Tuning & Ensembling (`src/tune.py`):** Tuned `ElasticNet` and `LGBMRegressor` across 5-fold CV using Optuna TPE (`40` trials each) and combined out-of-fold predictions (`65% ElasticNet + 35% LightGBM`).

## 5-Fold Cross-Validation Results

| Model | 5-Fold RMSLE | Notes |
| :--- | :---: | :--- |
| Dummy Regressor (Mean) | `0.3993 ± 0.0168` | Baseline benchmark |
| LightGBM (Default) | `0.1257 ± 0.0078` | Overfits small dataset (`num_leaves=31`) |
| Gradient Boosting (Default) | `0.1219 ± 0.0060` | Level-wise tree baseline |
| Ridge Regression (`alpha=10`) | `0.1142 ± 0.0073` | $L_2$ regularization |
| Lasso Regression (`alpha=0.0005`) | `0.1124 ± 0.0055` | $L_1$ sparsity drops noisy one-hot features |
| LightGBM (Optuna Tuned) | `0.1157` | Regularized (`num_leaves=12`, `colsample=0.41`) |
| ElasticNet (Optuna Tuned) | `0.1127` | Combined $L_1 + L_2$ penalty |
| **Blended Ensemble (65% ENet + 35% LGBM)** | **`0.1098`** | **Best OOF score; combines linear + tree signals** |