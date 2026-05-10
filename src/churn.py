"""Churn prediction with a Random Forest classifier."""

import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

CLEANED_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "cleaned.csv")
RFM_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "rfm.csv")
FEATIMP_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "feature_importance.csv")
PRED_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "churn_predictions.csv")

CHURN_DAYS = 90


def load_inputs():
    """Pull both the cleaned transactions and the rfm table."""
    if not os.path.exists(CLEANED_PATH):
        raise FileNotFoundError(f"{CLEANED_PATH} not found. Run preprocess.py first.")
    if not os.path.exists(RFM_PATH):
        raise FileNotFoundError(f"{RFM_PATH} not found. Run rfm.py first.")
    tx = pd.read_csv(CLEANED_PATH, parse_dates=["InvoiceDate"])
    rfm = pd.read_csv(RFM_PATH)
    return tx, rfm


def build_features(tx, rfm):
    """Per-customer feature matrix + churn label (1 = no purchase in last 90 days)."""
    snapshot = tx["InvoiceDate"].max()

    extras = tx.groupby("CustomerID").agg(
        TotalTransactions=("InvoiceNo", "count"),
        AvgOrderValue=("TotalPrice", "mean"),
        UniqueProducts=("StockCode", "nunique"),
    ).reset_index()

    df = rfm.merge(extras, on="CustomerID", how="left")
    df["Churn"] = (df["Recency"] > CHURN_DAYS).astype(int)

    feature_cols = [
        "R_Score", "F_Score", "M_Score",
        "TotalTransactions", "AvgOrderValue", "UniqueProducts",
    ]
    X = df[feature_cols]
    y = df["Churn"]
    return df, X, y, feature_cols, snapshot


def train(X, y, random_state=42):
    """Stratified 80/20 split, fit RF, return model + test scores + probabilities."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Classification report (test set):")
    print(classification_report(y_test, y_pred, digits=3))
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.3f}")

    return model, auc


def run():
    tx, rfm = load_inputs()
    df, X, y, feature_cols, snapshot = build_features(tx, rfm)

    print(f"Snapshot date  : {snapshot.date()}")
    print(f"Churn cutoff   : {CHURN_DAYS} days")
    print(f"Churn rate     : {y.mean():.1%}")
    print(f"Customers      : {len(df):,}\n")

    model, _ = train(X, y)

    # full-population predictions for the dashboard
    df["ChurnProbability"] = model.predict_proba(X)[:, 1]
    df["ChurnPrediction"] = model.predict(X)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    os.makedirs(os.path.dirname(FEATIMP_PATH), exist_ok=True)
    importance.to_csv(FEATIMP_PATH, index=False)
    df[["CustomerID", "Recency", "Frequency", "Monetary",
        "Churn", "ChurnPrediction", "ChurnProbability"]].to_csv(PRED_PATH, index=False)

    print(f"\nSaved feature importances -> {FEATIMP_PATH}")
    print(f"Saved predictions         -> {PRED_PATH}")
    return df, model


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
