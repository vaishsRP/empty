"""Streamlit dashboard for the retail customer intelligence project."""

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs")

st.set_page_config(page_title="Retail Customer Intelligence", layout="wide")


@st.cache_data
def load_csv(name, **kwargs):
    path = os.path.join(OUTPUTS, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, **kwargs)


def require(df, name, hint):
    if df is None:
        st.warning(f"Missing `{name}`. {hint}")
        st.stop()


tx = load_csv("cleaned.csv", parse_dates=["InvoiceDate"])
require(tx, "outputs/cleaned.csv", "Run `python src/preprocess.py` first.")

segments = load_csv("segments.csv")
predictions = load_csv("churn_predictions.csv")
feat_imp = load_csv("feature_importance.csv")
cohort = load_csv("cohort_retention.csv", index_col=0)

st.title("Retail Customer Intelligence")
st.caption("RFM segmentation, churn prediction, and cohort retention on the UCI Online Retail dataset.")

tab_overview, tab_rfm, tab_churn, tab_cohort = st.tabs(
    ["Overview", "RFM Segmentation", "Churn Risk", "Cohort Retention"]
)


with tab_overview:
    total_customers = tx["CustomerID"].nunique()
    total_revenue = tx["TotalPrice"].sum()
    avg_order = tx.groupby("InvoiceNo")["TotalPrice"].sum().mean()
    date_range = f"{tx['InvoiceDate'].min().date()} → {tx['InvoiceDate'].max().date()}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{total_customers:,}")
    c2.metric("Revenue", f"£{total_revenue:,.0f}")
    c3.metric("Avg order value", f"£{avg_order:,.2f}")
    c4.metric("Date range", date_range)

    st.subheader("Revenue by country (top 10)")
    by_country = (
        tx.groupby("Country")["TotalPrice"].sum()
        .sort_values(ascending=False).head(10).reset_index()
    )
    fig = px.bar(by_country, x="Country", y="TotalPrice", labels={"TotalPrice": "Revenue"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly revenue")
    monthly = tx.set_index("InvoiceDate")["TotalPrice"].resample("MS").sum().reset_index()
    fig = px.line(monthly, x="InvoiceDate", y="TotalPrice", labels={"TotalPrice": "Revenue"})
    st.plotly_chart(fig, use_container_width=True)


with tab_rfm:
    require(segments, "outputs/segments.csv", "Run `python src/segmentation.py` first.")

    st.subheader("Recency vs Monetary by segment")
    fig = px.scatter(
        segments,
        x="Recency", y="Monetary",
        color="Segment", size="Frequency",
        hover_data=["CustomerID", "Frequency"],
        size_max=30,
    )
    fig.update_layout(yaxis_type="log")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Customers per segment")
        counts = segments["Segment"].value_counts().reset_index()
        counts.columns = ["Segment", "Customers"]
        fig = px.bar(counts, x="Segment", y="Customers")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Average revenue per segment")
        avg = segments.groupby("Segment")["Monetary"].mean().reset_index()
        fig = px.bar(avg, x="Segment", y="Monetary", labels={"Monetary": "Avg revenue"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment guide")
    st.markdown(
        "- **Champions** — recent, frequent, high spend. Reward and retain.\n"
        "- **At Risk** — used to spend a lot but haven't returned recently. Win-back campaigns.\n"
        "- **Promising** — newer customers with limited history. Nurture into Champions.\n"
        "- **Lost** — long gone, low spend. Low-cost reactivation only."
    )


with tab_churn:
    require(predictions, "outputs/churn_predictions.csv", "Run `python src/churn.py` first.")
    require(feat_imp, "outputs/feature_importance.csv", "Run `python src/churn.py` first.")

    churn_rate = predictions["Churn"].mean()
    st.metric("Overall churn rate", f"{churn_rate:.1%}")

    st.subheader("Feature importance")
    fig = px.bar(
        feat_imp.sort_values("importance"),
        x="importance", y="feature", orientation="h",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 20 highest churn-risk customers")
    top = predictions.sort_values("ChurnProbability", ascending=False).head(20)
    top = top[["CustomerID", "Recency", "Frequency", "Monetary", "ChurnProbability"]]
    top["ChurnProbability"] = top["ChurnProbability"].round(3)
    st.dataframe(top, use_container_width=True, hide_index=True)


with tab_cohort:
    require(cohort, "outputs/cohort_retention.csv", "Run `python src/cohort.py` first.")

    st.subheader("Cohort retention (%)")
    matrix = cohort.copy()
    matrix.columns = matrix.columns.astype(str)

    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.index,
        colorscale="Blues",
        text=matrix.round(1).values,
        texttemplate="%{text}",
        hovertemplate="Cohort %{y}<br>Month %{x}<br>Retention %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Months since first purchase",
        yaxis_title="Cohort",
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)
