import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error

from src.features import HousePriceFeatureEngineer
from src.train import build_preprocessor

# Silence verbose trial-by-trial logs from Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_clean_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    # Drop the 2 extreme partial-sale outliers spotted during EDA
    train = train[~((train["GrLivArea"] > 4500) & (train["SalePrice"] < 300000))].reset_index(drop=True)

    X = train.drop(columns=["SalePrice"])
    y = np.log1p(train["SalePrice"])
    return X, y, test


def tune_elasticnet(X, y, cv) -> dict:
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.05, 1.0)

        pipe = Pipeline([
            ("features", HousePriceFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=15000, random_state=42)),
        ])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        return -scores.mean()

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=40)
    print(f"Best ElasticNet RMSLE: {study.best_value:.4f} | Params: {study.best_params}")
    return study.best_params


def tune_lightgbm(X, y, cv) -> dict:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 40),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "verbose": -1,
        }

        pipe = Pipeline([
            ("features", HousePriceFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", LGBMRegressor(**params)),
        ])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        return -scores.mean()

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=40)
    print(f"Best LightGBM   RMSLE: {study.best_value:.4f} | Params: {study.best_params}")
    return study.best_params


def main():
    X, y, test = load_clean_data()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print("Running Optuna tuning (40 trials per model)...")
    best_enet_params = tune_elasticnet(X, y, cv)
    best_lgbm_params = tune_lightgbm(X, y, cv)

    # Build tuned pipelines
    enet_pipe = Pipeline([
        ("features", HousePriceFeatureEngineer()),
        ("preprocess", build_preprocessor()),
        ("model", ElasticNet(**best_enet_params, max_iter=15000, random_state=42)),
    ])

    lgbm_pipe = Pipeline([
        ("features", HousePriceFeatureEngineer()),
        ("preprocess", build_preprocessor()),
        ("model", LGBMRegressor(**best_lgbm_params, subsample_freq=1, random_state=42, verbose=-1)),
    ])

    # Evaluate a 65% ElasticNet + 35% LightGBM out-of-fold blend
    oof_enet = cross_val_predict(enet_pipe, X, y, cv=cv, n_jobs=-1)
    oof_lgbm = cross_val_predict(lgbm_pipe, X, y, cv=cv, n_jobs=-1)
    oof_blend = 0.65 * oof_enet + 0.35 * oof_lgbm

    blend_rmsle = root_mean_squared_error(y, oof_blend)
    print(f"\nBlended OOF RMSLE (65% ElasticNet + 35% LightGBM): {blend_rmsle:.4f}")

    # Fit both tuned pipelines on 100% of clean training data & invert log1p -> real dollars
    enet_pipe.fit(X, y)
    lgbm_pipe.fit(X, y)

    test_log_preds = 0.65 * enet_pipe.predict(test) + 0.35 * lgbm_pipe.predict(test)
    test_dollar_preds = np.expm1(test_log_preds)

    submission = pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": test_dollar_preds
    })
    submission.to_csv("data/submission.csv", index=False)
    print(f"Saved data/submission.csv successfully! Rows: {len(submission)}")
    print(submission.head())


if __name__ == "__main__":
    main()