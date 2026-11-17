import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

from src.features import TitanicFeatureEngineer
from src.train import build_preprocessor

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X = train.drop(columns=["Survived"])
y = train["Survived"]

pipe = Pipeline([
        ("features", TitanicFeatureEngineer()),
        ("preprocess", build_preprocessor()),
        ("model", GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)),
    ])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

analysis_df = train[["PassengerId", "Name", "Sex", "Pclass", "Age", "Fare", "Survived"]].copy()
analysis_df["PredProb"] = oof_probs.round(3)
analysis_df["Error"] = (analysis_df["Survived"] - analysis_df["PredProb"]).abs()

print("\n=== Top 10 Worst Prediction Errors (Most Confident Mistakes) ===")
worst_errors = analysis_df.sort_values(by="Error", ascending=False).head(10)
print(worst_errors.to_string(index=False))

pipe.fit(X, y)
test_preds = pipe.predict(test)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": test_preds
    })
submission.to_csv("data/submission.csv", index=False)
print("\nSaved data/submission.csv successfully! Rows:", len(submission))