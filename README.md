# Retail Customer Intelligence

Online retailers spend most of their marketing budget acquiring customers and very little understanding the ones they already have. This project takes a year of UK online-retail transactions (UCI ML Repo dataset 352) and turns it into a small intelligence layer: who the best customers are, who is about to leave, and how well each acquisition cohort retains over time. The goal is a dashboard a non-technical operator can open and act on, not a research notebook.

> **At a glance**
> **Question:** with nothing but a transaction log — no demographics, no
> surveys — how much can a small retailer know about who to retain, who
> to win back, and whether retention is getting better or worse?
> **Approach:** RFM scoring → KMeans segmentation into four
> operator-memorable segments → a Random Forest churn model on top of the
> RFM features → monthly cohort retention. Deployed as a Streamlit
> dashboard over 540K transactions.
> **Decisions this supports:** which segment gets the win-back budget
> (At Risk, not Lost), which "Champions" are quietly going cold despite a
> high lifetime value, and whether this quarter's acquisition cohorts
> retain worse than last year's — the early warning that growth is
> masking a retention problem.
> **Honest caveat:** churn here means "90 days of silence," which is a
> proxy, not a fact. The limitations section explains why, and what would
> need to change in production.

## Components

**Preprocessing** drops the obvious noise — null `CustomerID`, returns (`Quantity <= 0`), zero-priced rows, and any country with fewer than ten transactions, which in this dataset is almost always a single mis-coded record. Everything downstream assumes a clean transaction-per-row table with a `TotalPrice` column.

**RFM** (Recency, Frequency, Monetary) is the standard starting point for transactional retail data because it captures the three things that matter for repeat-purchase behaviour without needing any demographic information. Scores are quintile-based (1–5 per dimension) so they are robust to long tails. The snapshot date is fixed at `max(InvoiceDate) + 1 day` so recency is meaningful even with a static historical extract.

**Segmentation** uses KMeans with `k=4` on scaled RFM values. Four was chosen deliberately: it lines up with the canonical retail segments (Champions / At Risk / Promising / Lost), it is small enough that an operator can remember what each group means, and silhouette scores on this dataset don't meaningfully improve past four. Cluster-to-label mapping is computed from centroid characteristics rather than hard-coded, so it stays correct even if KMeans returns clusters in a different order.

**Churn** is defined as "no purchase in the last 90 days of the dataset window." Ninety days is a defensible cutoff for a non-subscription retailer where typical reorder cadence is monthly — short enough that a churned customer is meaningfully gone, long enough not to flag normal seasonal gaps. Features are the RFM scores plus total transactions, average order value, and unique product count. A Random Forest is used because it handles the mixed-scale features and skewed target without much tuning. The model is evaluated with both a classification report and ROC-AUC because the classes are imbalanced.

**Cohort retention** assigns each customer to their first-purchase month and computes the percentage of that cohort still active in each subsequent month. This is the standard view for spotting whether retention is improving or degrading over time.

## Install and run

```bash
pip install -r requirements.txt

python src/preprocess.py
python src/rfm.py
python src/segmentation.py
python src/churn.py
python src/cohort.py

streamlit run app.py
```

The data file (`data/online_retail.xlsx`) ships with the repo. If you want to refetch it from source:

```python
from ucimlrepo import fetch_ucirepo
fetch_ucirepo(id=352).data.features.to_excel("data/online_retail.xlsx", index=False)
```

## Limitations and next steps

- The dataset is a single ~13-month window from one UK retailer, so the churn cutoff and cohort patterns reflect that specific business cycle. Re-tuning would be needed for a different vertical or a multi-year window.
- Churn is defined purely by inactivity. We have no signal for customers who explicitly cancelled, switched to a different channel, or were lost to a refund dispute, so the label is noisy by construction.
- Segmentation is unsupervised and recomputed from scratch each run. For production use you would freeze the scaler and cluster centroids, otherwise customers will drift between segments simply because the model was refit, not because their behaviour changed.
