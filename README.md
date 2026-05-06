# Retail Customer Intelligence

Online retailers spend most of their marketing budget acquiring customers and very little understanding the ones they already have. This project takes a year of UK online-retail transactions (UCI M[...]

## Components

**Preprocessing** drops the obvious noise — null `CustomerID`, returns (`Quantity <= 0`), zero-priced rows, and any country with fewer than ten transactions, which in this dataset is almost alway[...]

**RFM** (Recency, Frequency, Monetary) is the standard starting point for transactional retail data because it captures the three things that matter for repeat-purchase behaviour without needing an[...]

**Segmentation** uses KMeans with `k=4` on scaled RFM values. Four was chosen deliberately: it lines up with the canonical retail segments (Champions / At Risk / Promising / Lost), it is small eno[...]

**Churn** is defined as "no purchase in the last 90 days of the dataset window." Ninety days is a defensible cutoff for a non-subscription retailer where typical reorder cadence is monthly — sho[...]

**Cohort retention** assigns each customer to their first-purchase month and computes the percentage of that cohort still active in each subsequent month. This is the standard view for spotting wh[...]

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

- The dataset is a single ~13-month window from one UK retailer, so the churn cutoff and cohort patterns reflect that specific business cycle. Re-tuning would be needed for a different vertical or[...]
- Churn is defined purely by inactivity. We have no signal for customers who explicitly cancelled, switched to a different channel, or were lost to a refund dispute, so the label is noisy by const[...]
- Segmentation is unsupervised and recomputed from scratch each run. For production use you would freeze the scaler and cluster centroids, otherwise customers will drift between segments simply be[...]
