"""
Trains the Logistic Regression and Random Forest pipelines and saves them
to disk (models/) for the Streamlit app to load, plus a small metadata.json
of evaluation metrics and sample flagged transactions so the app doesn't
need to carry the full 61k-row dataset around.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)

from features import engineer_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES

DATA_PATH = "transactions.csv"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH, dtype={"mcc": str})
    df = engineer_features(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    lr_pipe = Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    rf_pipe = Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1)),
    ])

    lr_pipe.fit(X_train, y_train)
    rf_pipe.fit(X_train, y_train)

    metrics = {}
    for name, pipe in [("logistic_regression", lr_pipe), ("random_forest", rf_pipe)]:
        proba = pipe.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        cm = confusion_matrix(y_test, preds)
        metrics[name] = {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "pr_auc": float(average_precision_score(y_test, proba)),
            "confusion_matrix": {
                "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
            },
        }

    joblib.dump(lr_pipe, MODELS_DIR / "lr_pipeline.joblib")
    joblib.dump(rf_pipe, MODELS_DIR / "rf_pipeline.joblib")

    # feature importance from the RF
    ohe = rf_pipe.named_steps["prep"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_names = NUMERIC_FEATURES + cat_names
    importances = rf_pipe.named_steps["clf"].feature_importances_
    imp = sorted(zip(all_names, importances.tolist()), key=lambda t: -t[1])[:8]

    # a handful of real flagged test-set transactions for the demo table
    test_df = df.loc[X_test.index]
    flagged = test_df[test_df["is_fraud"] == 1].sample(6, random_state=7)
    flagged_records = flagged[
        ["transaction_id", "card_id", "channel", "amount", "currency",
         "fraud_pattern", "city", "hour"]
    ].to_dict(orient="records")

    metadata = {
        "n_transactions": int(len(df)),
        "fraud_rate": float(df["is_fraud"].mean()),
        "test_size": int(len(X_test)),
        "metrics": metrics,
        "feature_importance": imp,
        "flagged_samples": flagged_records,
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata["metrics"], indent=2))
    print(f"\nSaved models to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
