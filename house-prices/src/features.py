import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class HousePriceFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lot = None
        self.glo_lot = None
        self.med = {}
        self.cat_mod = {}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()

        if "Id" in x.columns:
            x = x.drop(columns=["Id"])

        # Categorical columns where NaN means "None" (feature absent)
        none_cat = [
            "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
            "GarageType", "GarageFinish", "GarageQual", "GarageCond",
            "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
            "MasVnrType"
        ]
        for col in none_cat:
            if col in x.columns:
                x[col] = x[col].fillna("None")

        # Numerical columns where NaN means 0 (no garage, basement, or veneer)
        zero_num_cols = [
            "GarageYrBlt", "GarageArea", "GarageCars",
            "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
            "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
        ]
        for col in zero_num_cols:
            if col in x.columns:
                x[col] = x[col].fillna(0)

        # Map ordered quality ratings (Ex/Gd/TA/Fa/Po/None) to 5..0
        qual_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
        qual_cols = [
            "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
            "HeatingQC", "KitchenQual", "FireplaceQu",
            "GarageQual", "GarageCond", "PoolQC"
        ]
        for col in qual_cols:
            if col in x.columns:
                x[col] = x[col].map(qual_map).fillna(0)

        # MSSubClass is a dwelling type code, not a continuous number
        if "MSSubClass" in x.columns:
            x["MSSubClass"] = x["MSSubClass"].astype(str)

        return x

    def _add_domain_features(self, X: pd.DataFrame) -> pd.DataFrame:
        # Total square footage across basement + 1st + 2nd floor
        X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]

        # Total bathrooms (half baths weighted at 0.5)
        X["TotalBath"] = (
            X["FullBath"] + 0.5 * X["HalfBath"]
            + X["BsmtFullBath"] + 0.5 * X["BsmtHalfBath"]
        )

        # Total porch & deck area
        X["TotalPorchSF"] = (
            X["OpenPorchSF"] + X["EnclosedPorch"]
            + X["3SsnPorch"] + X["ScreenPorch"] + X["WoodDeckSF"]
        )

        # House age and years since remodel at time of sale
        X["HouseAge"] = (X["YrSold"] - X["YearBuilt"]).clip(lower=0)
        X["RemodAge"] = (X["YrSold"] - X["YearRemodAdd"]).clip(lower=0)
        X["HasPool"] = (X["PoolArea"] > 0).astype(int)
        X["HasGarage"] = (X["GarageArea"] > 0).astype(int)

        return X

    def fit(self, X: pd.DataFrame, y=None):
        X_clean = self.clean(X)

        self.lot = X_clean.groupby("Neighborhood")["LotFrontage"].median()
        self.glo_lot = X_clean["LotFrontage"].median()

        numcol = X_clean.select_dtypes(include=[np.number]).columns
        catcol = X_clean.select_dtypes(exclude=[np.number]).columns

        self.med = {col: X_clean[col].median() for col in numcol}
        self.cat_mod = {col: X_clean[col].mode()[0] for col in catcol}

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_clean = self.clean(X)

        # Fill missing LotFrontage by Neighborhood median learned in .fit()
        for neighborhood, med in self.lot.items():
            mask = X_clean["LotFrontage"].isnull() & (X_clean["Neighborhood"] == neighborhood)
            X_clean.loc[mask, "LotFrontage"] = med
        X_clean["LotFrontage"] = X_clean["LotFrontage"].fillna(self.glo_lot)

        # Fill any remaining NaNs using training medians/modes
        for col, med in self.med.items():
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].fillna(med)

        for col, mode_val in self.cat_mod.items():
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].fillna(mode_val)

        # Engineer combined domain features after all base columns are imputed
        X_clean = self._add_domain_features(X_clean)

        return X_clean