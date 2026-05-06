"""Compute RFM (Recency, Frequency, Monetary) metrics per customer."""

import os
import sys
import pandas as pd

CLEANED_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "cleaned.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "rfm.csv")


def load_cleaned(path=CLEANED_PATH):
    """Load the cleaned transactions CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run preprocess.py first."
        )
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    return df


def compute_rfm(df):
    """Aggregate per-customer R, F, M values. Snapshot = max date + 1 day."""
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    return rfm, snapshot


def score_rfm(rfm):
    """Score each dimension 1-5 by quintile. 5 = best."""
    # recency: lower is better, so reverse the labels
    rfm["R_Score"] = pd.qcut(rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )
    return rfm


def run():
    df = load_cleaned()
    rfm, snapshot = compute_rfm(df)
    rfm = score_rfm(rfm)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    rfm.to_csv(OUTPUT_PATH, index=False)

    print(f"Snapshot date : {snapshot.date()}")
    print(f"Customers     : {len(rfm):,}")
    print(rfm.head())
    print(f"Saved RFM table -> {OUTPUT_PATH}")
    return rfm


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
