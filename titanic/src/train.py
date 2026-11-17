import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from src.features import TitanicFeatureEngineer


def build_preprocessor() -> ColumnTransformer:
    num_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "HasCabin", "FarePerPerson"]
    cat_cols = ["Sex", "Embarked", "Title"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )


def main():
    train = pd.read_csv("data/train.csv")
    X = train.drop(columns=["Survived"])
    y = train["Survived"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Dummy (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42),
        "SVM (RBF Kernel)": SVC(C=1.0, probability=True, random_state=42),
    }

    scoring = ["accuracy", "f1", "roc_auc"]
    results = []

    for name, model in models.items():
        # Every fold runs feature engineering + scaling/encoding strictly inside the training split
        pipe = Pipeline([
            ("features", TitanicFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", model),
        ])

        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)

        results.append({
            "Model": name,
            "Accuracy": f"{scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}",
            "F1-Score": f"{scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}",
            "ROC-AUC": f"{scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}",
        })

    results_df = pd.DataFrame(results)
    print("\n=== 5-Fold Stratified Cross-Validation Results ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()