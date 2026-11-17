# Titanic Survival Prediction — Leak-Free Classical ML Pipeline

End-to-end binary classification pipeline predicting passenger survival on the Kaggle Titanic dataset, built with strict cross-validation discipline and zero data leakage.

## Key Engineering Highlights
- **Leak-Free Custom Transformer (`src/features.py`):** Built a `scikit-learn` compatible `BaseEstimator` / `TransformerMixin` class so all group medians (`Age` imputed by `(Pclass, Title)`) and modes are computed strictly inside each training fold.
- **Hypothesis-Driven Features:** Extracted passenger `Title`, `FamilySize`, `IsAlone`, `HasCabin` (missingness indicator), and `FarePerPerson` based on Exploratory Data Analysis (`notebooks/01_eda.ipynb`).
- **Stratified 5-Fold Cross-Validation (`src/train.py`):** Preserved the 38.4% / 61.6% class distribution across folds and evaluated models using Accuracy, F1-Score, and ROC-AUC.

## 5-Fold Stratified Cross-Validation Results

| Model | Accuracy | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: |
| Dummy (Most Frequent) | 0.6162 ± 0.0023 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |
| Logistic Regression | 0.8305 ± 0.0084 | 0.7746 ± 0.0163 | 0.8730 ± 0.0201 |
| Random Forest | 0.8361 ± 0.0099 | 0.7729 ± 0.0183 | 0.8745 ± 0.0190 |
| **Gradient Boosting** | **0.8395 ± 0.0143** | **0.7824 ± 0.0183** | **0.8799 ± 0.0154** |
| SVM (RBF Kernel) | 0.8283 ± 0.0107 | 0.7659 ± 0.0195 | 0.8619 ± 0.0214 |

## Error Analysis Takeaways (`src/evaluate.py`)
Inspecting out-of-fold probability errors revealed that the model's highest-confidence mistakes were rare historical anomalies—such as 1st-class women and children (e.g., the Allison family) who perished despite >94% predicted survival probability, and 3rd-class adult men who survived against <5% odds.

## How to Run
```bash
python -m venv venv
venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
python -m src.train
python -m src.evaluate