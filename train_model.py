"""
Fraud Detection Model - Feature Engineering + Training + Evaluation
=====================================================================
Trains a classifier to detect fraudulent transactions on the synthetic
payment-switch dataset (transactions.csv).

Approach:
  1. Engineer per-card behavioral features (velocity, amount deviation,
     geo movement, time-of-day) computed causally (using only prior
     transaction history at each point in time - no leakage).
  2. Train a Random Forest and a Logistic Regression baseline.
  3. Evaluate with metrics appropriate for imbalanced fraud data:
     precision, recall, F1, ROC-AUC, PR-AUC, and a confusion matrix
     at a chosen operating threshold.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve
)

DATA_PATH = "/home/claude/zimswitch_fraud/transactions.csv"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def engineer_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["card_id", "timestamp"]).reset_index(drop=True)

    df["hour"] = df["timestamp"].dt.hour
    df["is_odd_hour"] = df["hour"].between(1, 4).astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Per-card rolling behavioral stats, computed causally (shift(1) so the
    # current transaction never sees its own value)
    grp = df.groupby("card_id")

    df["prev_timestamp"] = grp["timestamp"].shift(1)
    df["seconds_since_prev"] = (
        (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds()
    )
    df["seconds_since_prev"] = df["seconds_since_prev"].fillna(999999)

    df["prev_lat"] = grp["latitude"].shift(1)
    df["prev_lon"] = grp["longitude"].shift(1)
    df["km_from_prev_txn"] = haversine_km(
        df["latitude"], df["longitude"], df["prev_lat"], df["prev_lon"]
    ).fillna(0)
    # implied speed (km/h) between consecutive transactions - a hallmark
    # "impossible travel" fraud signal
    hours_gap = (df["seconds_since_prev"] / 3600).replace(0, 1 / 3600)
    df["implied_speed_kmh"] = (df["km_from_prev_txn"] / hours_gap).clip(upper=2000)

    # expanding mean/std of amount per card, using only prior transactions
    # (computed manually with cumulative sums to avoid groupby/apply index
    # alignment issues, and to keep this fast on tens of thousands of rows)
    shifted = grp["amount"].shift(1)
    df["_shifted_amount"] = shifted
    df["_has_prior"] = df["_shifted_amount"].notna().astype(int)
    df["_shifted_filled"] = df["_shifted_amount"].fillna(0)
    df["_shifted_filled_sq"] = df["_shifted_filled"] ** 2

    g2 = df.groupby("card_id")
    cum_count = g2["_has_prior"].cumsum()
    cum_sum = g2["_shifted_filled"].cumsum()
    cum_sumsq = g2["_shifted_filled_sq"].cumsum()

    df["running_avg_amount"] = (cum_sum / cum_count.replace(0, np.nan))
    running_var = (cum_sumsq / cum_count.replace(0, np.nan)) - df["running_avg_amount"] ** 2
    df["running_std_amount"] = np.sqrt(running_var.clip(lower=0))

    df = df.drop(columns=["_has_prior", "_shifted_filled", "_shifted_filled_sq"])

    df["running_avg_amount"] = df["running_avg_amount"].fillna(df["amount"].median())
    df["running_std_amount"] = df["running_std_amount"].fillna(0).replace(0, 1)
    df["amount_zscore"] = (df["amount"] - df["running_avg_amount"]) / df["running_std_amount"]
    df = df.drop(columns=["_shifted_amount"])

    # transaction count in the trailing 1 hour and 24 hours per card
    def rolling_count(sub, window):
        sub = sub.set_index("timestamp")
        counts = sub["amount"].rolling(window, closed="left").count()
        return counts.values

    df["txn_count_1h"] = 0
    df["txn_count_24h"] = 0
    for card_id, sub in df.groupby("card_id"):
        idx = sub.index
        df.loc[idx, "txn_count_1h"] = rolling_count(sub, "1h")
        df.loc[idx, "txn_count_24h"] = rolling_count(sub, "24h")
    df["txn_count_1h"] = df["txn_count_1h"].fillna(0)
    df["txn_count_24h"] = df["txn_count_24h"].fillna(0)

    return df


def main():
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    feature_cols_numeric = [
        "amount", "hour", "is_odd_hour", "day_of_week",
        "seconds_since_prev", "km_from_prev_txn", "implied_speed_kmh",
        "amount_zscore", "txn_count_1h", "txn_count_24h",
    ]
    feature_cols_categorical = ["channel", "mcc", "currency"]

    X = df[feature_cols_numeric + feature_cols_categorical]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), feature_cols_numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_categorical),
    ])

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        ),
    }

    results = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocess), ("clf", model)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        report = classification_report(y_test, preds, digits=3)
        cm = confusion_matrix(y_test, preds)

        results[name] = dict(pipe=pipe, roc_auc=roc_auc, pr_auc=pr_auc,
                              report=report, cm=cm, proba=proba)

        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        print(f"ROC-AUC: {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}")
        print(report)
        print("Confusion matrix [[TN, FP], [FN, TP]]:")
        print(cm)

    # feature importance from the Random Forest
    rf_pipe = results["Random Forest"]["pipe"]
    ohe = rf_pipe.named_steps["prep"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(feature_cols_categorical))
    all_feature_names = feature_cols_numeric + cat_feature_names
    importances = rf_pipe.named_steps["clf"].feature_importances_
    imp_df = pd.DataFrame({
        "feature": all_feature_names, "importance": importances
    }).sort_values("importance", ascending=False).head(15)

    print(f"\n{'=' * 60}\nTop 15 Random Forest Feature Importances\n{'=' * 60}")
    print(imp_df.to_string(index=False))

    imp_df.to_csv("/home/claude/zimswitch_fraud/feature_importance.csv", index=False)

    summary = pd.DataFrame({
        "model": list(results.keys()),
        "roc_auc": [results[m]["roc_auc"] for m in results],
        "pr_auc": [results[m]["pr_auc"] for m in results],
    })
    summary.to_csv("/home/claude/zimswitch_fraud/model_comparison.csv", index=False)
    print(f"\nSaved feature_importance.csv and model_comparison.csv")


if __name__ == "__main__":
    main()
