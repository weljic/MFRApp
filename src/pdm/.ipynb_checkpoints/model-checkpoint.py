# src/pdm/model.py
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier  # or RandomForest, etc.
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)

from .features import build_preprocessor

def build_model():
    preprocessor = build_preprocessor()

    clf = GradientBoostingClassifier(
        random_state=42
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", clf),
        ]
    )
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print("ROC AUC:", roc_auc)
    print("PR AUC:", pr_auc)
    print(classification_report(y_test, y_pred))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }