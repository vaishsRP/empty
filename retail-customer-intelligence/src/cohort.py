"""Cohort retention analysis by month of first purchase."""

import os
import sys
import pandas as pd

CLEANED_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "cleaned.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "cohort_retention.csv")


def load_cleaned(path=CLEANED_PATH):
    """Read the cleaned transactions CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run preprocess.py first.")
    return pd.read_csv(path, parse_dates=["InvoiceDate"])


def _month_period(s):
    return s.dt.to_period("M")


def build_cohorts(df):
    """For each customer assign their acquisition (first-purchase) month."""
    df = df.copy()
    df["InvoiceMonth"] = _month_period(df["InvoiceDate"])
    df["CohortMonth"] = df.groupby("CustomerID")["InvoiceDate"].transform("min")
    df["CohortMonth"] = _month_period(df["CohortMonth"])
    return df


def retention_matrix(df):
    """Return a cohort x period matrix of retention % (rows = cohorts)."""
    df = build_cohorts(df)

    # period offset = months between cohort and current invoice
    df["PeriodIndex"] = (
        (df["InvoiceMonth"].dt.year - df["CohortMonth"].dt.year) * 12
        + (df["InvoiceMonth"].dt.month - df["CohortMonth"].dt.month)
    )

    grouped = df.groupby(["CohortMonth", "PeriodIndex"])["CustomerID"].nunique().reset_index()
    counts = grouped.pivot(index="CohortMonth", columns="PeriodIndex", values="CustomerID")

    cohort_sizes = counts.iloc[:, 0]
    retention = counts.divide(cohort_sizes, axis=0) * 100

    retention.index = retention.index.astype(str)
    return retention.round(1)


def run():
    df = load_cleaned()
    matrix = retention_matrix(df)

    print("Cohort retention matrix (% of cohort active in month N):")
    print(matrix.fillna("").to_string())

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    matrix.to_csv(OUTPUT_PATH)
    print(f"\nSaved cohort matrix -> {OUTPUT_PATH}")
    return matrix


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
