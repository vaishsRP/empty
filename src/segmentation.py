"""KMeans segmentation on RFM values, with business-readable labels."""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

RFM_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "rfm.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "segments.csv")

K = 4


def load_rfm(path=RFM_PATH):
    """Load the rfm table produced by rfm.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run rfm.py first.")
    return pd.read_csv(path)


def cluster(rfm, k=K, random_state=42):
    """Scale RFM values, fit KMeans, attach cluster column, return silhouette."""
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    rfm["Cluster"] = km.fit_predict(X)

    sil = silhouette_score(X, rfm["Cluster"])
    return rfm, km, scaler, sil


def label_clusters(rfm):
    """Map cluster ids -> business names by looking at centroid characteristics."""
    centroids = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

    # rank each cluster on each dimension
    centroids["r_rank"] = centroids["Recency"].rank()       # 1 = lowest recency = best
    centroids["f_rank"] = centroids["Frequency"].rank(ascending=False)  # 1 = highest freq = best
    centroids["m_rank"] = centroids["Monetary"].rank(ascending=False)   # 1 = highest spend = best

    labels = {}
    used = set()

    # champions: best F and M, low R
    champ = (centroids["f_rank"] + centroids["m_rank"] + centroids["r_rank"]).idxmin()
    labels[champ] = "Champions"
    used.add(champ)

    # at risk: high M, high R (used to spend, haven't returned)
    remaining = centroids.drop(index=list(used))
    at_risk = (remaining["m_rank"] - remaining["r_rank"]).idxmin()
    labels[at_risk] = "At Risk"
    used.add(at_risk)

    # lost: high R, low F, low M
    remaining = centroids.drop(index=list(used))
    lost = (remaining["r_rank"] * -1 + remaining["f_rank"] + remaining["m_rank"]).idxmax()
    labels[lost] = "Lost"
    used.add(lost)

    # promising: whatever's left (newer customers, low R, low F)
    remaining = centroids.drop(index=list(used))
    for idx in remaining.index:
        labels[idx] = "Promising"

    rfm["Segment"] = rfm["Cluster"].map(labels)
    return rfm, labels


def run():
    rfm = load_rfm()
    rfm, _, _, sil = cluster(rfm)
    rfm, labels = label_clusters(rfm)

    print(f"Silhouette score (k={K}): {sil:.3f}")
    print("Cluster -> Segment mapping:")
    for k, v in labels.items():
        print(f"  {k} -> {v}")
    print()
    print(rfm.groupby("Segment").agg(
        customers=("CustomerID", "count"),
        avg_recency=("Recency", "mean"),
        avg_frequency=("Frequency", "mean"),
        avg_monetary=("Monetary", "mean"),
    ).round(2))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    rfm.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved segments -> {OUTPUT_PATH}")
    return rfm


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
