"""Load and clean the Online Retail transactional data."""

import os
import sys
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "online_retail.xlsx")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "cleaned.csv")


def load_raw(path=DATA_PATH):
    """Read the source xlsx and parse InvoiceDate as datetime."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Place online_retail.xlsx inside the data/ folder."
        )
    try:
        df = pd.read_excel(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df


def clean(df):
    """Drop nulls, returns, bad prices, and tiny-country outliers. Add TotalPrice."""
    original = len(df)

    df = df.dropna(subset=["CustomerID"]).copy()
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    # drop countries with fewer than 10 rows — usually a single junk record
    counts = df["Country"].value_counts()
    rare = counts[counts < 10].index.tolist()
    if rare:
        df = df[~df["Country"].isin(rare)]

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    final = len(df)
    return df, original, final


def report(df, original, final):
    """Print a short data-quality report to stdout."""
    print("=" * 50)
    print("Data Quality Report")
    print("=" * 50)
    print(f"Original rows : {original:,}")
    print(f"Removed rows  : {original - final:,}")
    print(f"Final rows    : {final:,}")
    print(f"Date range    : {df['InvoiceDate'].min()} -> {df['InvoiceDate'].max()}")
    print(f"Customers     : {df['CustomerID'].nunique():,}")
    print(f"Countries     : {df['Country'].nunique()}")
    print("=" * 50)


def run():
    df = load_raw()
    df, original, final = clean(df)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    report(df, original, final)
    print(f"Saved cleaned data -> {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
