import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor

from src.features import HousePriceFeatureEngineer


def build_preprocessor() -> ColumnTransformer:
    # RobustScaler uses median & IQR so remaining numeric outliers don't distort scaling
    num_selector = make_column_selector(dtype_include=np.number)
    cat_selector = make_column_selector(dtype_exclude=np.number)

    return ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), num_selector),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_selector),
        ]
    )


def main():
    train = pd.read_csv("data/train.csv")

    # Remove the 2 extreme partial-sale outliers spotted during EDA
    train = train[~((train["GrLivArea"] > 4500) & (train["SalePrice"] < 300000))].reset_index(drop=True)

    X = train.drop(columns=["SalePrice"])
    y = np.log1p(train["SalePrice"])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Dummy (Mean Baseline)": DummyRegressor(strategy="mean"),
        "Ridge Regression": Ridge(alpha=10.0),
        "Lasso Regression": Lasso(alpha=0.0005, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=0.0005, l1_ratio=0.5, max_iter=10000),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1
        ),
    }

    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    results = []
    for name, model in models.items():
        pipe = Pipeline([
            ("features", HousePriceFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", model),
        ])

        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        # Flip sign on neg_rmse / neg_mae so lower positive values = better
        rmse_mean = -scores["test_rmse"].mean()
        rmse_std = scores["test_rmse"].std()
        mae_mean = -scores["test_mae"].mean()
        r2_mean = scores["test_r2"].mean()

        results.append({
            "Model": name,
            "RMSLE (Log-RMSE)": f"{rmse_mean:.4f} ± {rmse_std:.4f}",
            "Log-MAE": f"{mae_mean:.4f}",
            "R2 Score": f"{r2_mean:.4f}",
        })

    results_df = pd.DataFrame(results)
    print("\n=== 5-Fold Cross-Validation Baseline Results ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()