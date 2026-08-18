"""
train_model.py
Trains and compares three models for churn prediction, picks the best one
by ROC-AUC, and saves the winning pipeline (preprocessing + model) to disk
with joblib so the Streamlit app can load it directly.
"""
import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier


def main():
    df = pd.read_csv("../data/telco_clean.csv")

    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=42,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        ),
    }

    results = {}
    fitted_pipelines = {}

    for name, model in candidates.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        }
        fitted_pipelines[name] = pipe
        print(f"\n{name}")
        print(json.dumps(results[name], indent=2))

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_pipe = fitted_pipelines[best_name]
    print(f"\nBest model by ROC-AUC: {best_name}")

    # Confusion matrix + report for the winning model
    y_pred_best = best_pipe.predict(X_test)
    print("\nConfusion matrix (best model):")
    print(confusion_matrix(y_test, y_pred_best))
    print("\nClassification report (best model):")
    print(classification_report(y_test, y_pred_best))

    # Save the winning pipeline + metadata
    joblib.dump(best_pipe, "../model/churn_pipeline.joblib")

    with open("../model/results.json", "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    with open("../model/feature_columns.json", "w") as f:
        json.dump({
            "categorical_cols": categorical_cols,
            "numeric_cols": numeric_cols,
        }, f, indent=2)

    print("\nSaved model to model/churn_pipeline.joblib")
    print("Saved metrics to model/results.json")


if __name__ == "__main__":
    main()
