import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Store training-fold medians here during .fit() to prevent data leakage
        self.age_medians_ = None
        self.global_age_median_ = None
        self.fare_median_ = None
        self.embarked_mode_ = None

    def _extract_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()

        # 1. Extract Title and group rare titles into 'Rare'
        X["Title"] = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        rare_titles = [
            "Lady", "Countess", "Capt", "Col", "Don", "Dr",
            "Major", "Rev", "Sir", "Jonkheer", "Dona"
        ]
        X["Title"] = X["Title"].replace(rare_titles, "Rare")
        X["Title"] = X["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

        # 2. Family features
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)

        # 3. Cabin missingness signal
        X["HasCabin"] = X["Cabin"].notnull().astype(int)

        return X

    def fit(self, X: pd.DataFrame, y=None):
        X_feat = self._extract_base_features(X)

        # Learn medians/modes ONLY from the training fold
        self.age_medians_ = X_feat.groupby(["Pclass", "Title"])["Age"].median()
        self.global_age_median_ = X_feat["Age"].median()
        self.fare_median_ = X_feat["Fare"].median()
        self.embarked_mode_ = X_feat["Embarked"].mode()[0]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_feat = self._extract_base_features(X)

        # Fill missing Age using the (Pclass, Title) median learned in .fit()
        for (pclass, title), median_age in self.age_medians_.items():
            mask = (
                X_feat["Age"].isnull()
                & (X_feat["Pclass"] == pclass)
                & (X_feat["Title"] == title)
            )
            X_feat.loc[mask, "Age"] = median_age

        # Fallback if an unseen (Pclass, Title) combo has missing Age
        X_feat["Age"] = X_feat["Age"].fillna(self.global_age_median_)
        X_feat["Fare"] = X_feat["Fare"].fillna(self.fare_median_)
        X_feat["Embarked"] = X_feat["Embarked"].fillna(self.embarked_mode_)

        # Feature that depends on imputed Fare
        X_feat["FarePerPerson"] = X_feat["Fare"] / X_feat["FamilySize"]

        # Keep only the clean modeling columns (drop raw IDs, Name, Ticket, Cabin)
        keep_cols = [
            "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
            "Embarked", "Title", "FamilySize", "IsAlone", "HasCabin", "FarePerPerson"
        ]
        return X_feat[keep_cols]