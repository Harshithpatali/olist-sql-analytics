import io
import json
import html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.db import run_query, test_connection
from utils.llm import generate_stakeholder_brief, brief_to_markdown, explain_to_stakeholder

# ============================================================
# Configuration
# ============================================================
st.set_page_config(
    page_title="Olist Business Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Business assumptions — change these to your real operating targets.
ON_TIME_TARGET = 90.0
GOOD_RATING = 4.0
DEFAULT_MIN_MONTHLY_ORDERS = 100

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    border: 1px solid #e8eaed;
    border-radius: 12px;
    padding: 12px 14px;
    background: white;
}
.brief-card {
    border: 1px solid #e4e7eb; border-radius: 14px; padding: 20px;
    background: #ffffff; margin: 8px 0 18px 0;
}
.status-pill {
    display: inline-block; padding: 5px 11px; border-radius: 999px;
    font-weight: 700; font-size: 0.82rem; margin-bottom: 8px;
}
.status-strong {background:#e8f7ee;color:#167a45;}
.status-stable {background:#edf4ff;color:#2359a6;}
.status-watch {background:#fff5df;color:#946200;}
.status-risk {background:#ffe9e9;color:#a32626;}
.finding-number {font-size:1.65rem;font-weight:750;line-height:1.1;}
.finding-label {font-weight:650;margin-top:3px;}
.muted {color:#667085;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Data access
# ============================================================
@st.cache_data(ttl=3600)
def load_monthly_kpis():
    return run_query("SELECT * FROM vw_monthly_kpis ORDER BY purchase_month")

@st.cache_data(ttl=3600)
def load_rfm():
    return run_query("SELECT * FROM vw_rfm_segments")

@st.cache_data(ttl=3600)
def load_cohort():
    return run_query("SELECT * FROM vw_cohort_retention")

@st.cache_data(ttl=3600)
def load_product():
    return run_query("SELECT * FROM vw_product_performance ORDER BY revenue DESC LIMIT 30")

@st.cache_data(ttl=3600)
def load_seller():
    return run_query("SELECT * FROM vw_seller_performance ORDER BY revenue DESC LIMIT 50")

@st.cache_data(ttl=3600)
def load_state():
    return run_query("SELECT * FROM vw_state_performance")

@st.cache_data(ttl=3600)
def load_new_returning():
    return run_query("SELECT * FROM vw_new_vs_returning ORDER BY purchase_month")

@st.cache_data(ttl=3600)
def load_customer_metrics():
    return run_query("SELECT * FROM vw_customer_metrics")


def plot_chart(fig, **kwargs):
    """Use Streamlit's current width API, with compatibility for older versions."""
    try:
        st.plotly_chart(fig, width="stretch", **kwargs)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kwargs)


def show_df(df, **kwargs):
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


def filtered_kpis():
    kpis = load_monthly_kpis().copy()
    kpis["total_orders"] = pd.to_numeric(kpis["total_orders"], errors="coerce")
    return kpis[kpis["total_orders"] >= st.session_state.min_monthly_orders].copy()


def pct_change(current, previous):
    if previous in (0, None) or pd.isna(previous):
        return np.nan
    return (current - previous) / previous * 100


def status_from_metrics(numbers):
    """Deterministic status; the LLM does not decide the status."""
    checks = []
    if numbers.get("on_time_pct") is not None:
        checks.append(numbers["on_time_pct"] >= ON_TIME_TARGET)
    if numbers.get("avg_rating") is not None:
        checks.append(numbers["avg_rating"] >= GOOD_RATING)
    if numbers.get("returning_revenue_share") is not None:
        checks.append(numbers["returning_revenue_share"] >= 30)
    if numbers.get("m1_retention") is not None:
        checks.append(numbers["m1_retention"] >= 20)
    if not checks:
        return "Stable"
    score = sum(bool(x) for x in checks) / len(checks)
    if score >= .80:
        return "Strong"
    if score >= .50:
        return "Stable"
    if score >= .25:
        return "Watch closely"
    return "At risk"


def render_status(status):
    cls = {
        "Strong": "status-strong",
        "Stable": "status-stable",
        "Watch closely": "status-watch",
        "At risk": "status-risk",
    }.get(status, "status-stable")
    st.markdown(f'<span class="status-pill {cls}">{html.escape(status)}</span>', unsafe_allow_html=True)


def render_brief(brief, status, audience, context, numbers):
    render_status(status)
    st.markdown(f"### {brief.get('headline', 'Stakeholder conclusion')}")
    st.write(brief.get("summary", ""))

    findings = brief.get("findings", [])[:4]
    if findings:
        cols = st.columns(len(findings))
        for col, item in zip(cols, findings):
            with col:
                st.markdown(f'<div class="finding-number">{html.escape(str(item.get("number","")))}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="finding-label">{html.escape(str(item.get("label","")))}</div>', unsafe_allow_html=True)
                st.caption(item.get("detail", ""))

    st.markdown("**Risks**")
    for risk in brief.get("risks", []):
        st.markdown(f"- {risk}")

    st.markdown("**Prioritised actions**")
    for action in sorted(brief.get("actions", []), key=lambda x: x.get("priority", 99)):
        st.markdown(f"**{action.get('priority','')}. {action.get('action','')}**")
        st.markdown(f"- Reason: {action.get('reason','')}")
        st.markdown(f"- Expected impact: {action.get('expected_impact','')}")

    st.caption(f"**Caveat:** {brief.get('caveat', '')}")

    markdown = brief_to_markdown(brief, status, audience, context)
    st.download_button(
        "⬇️ Download brief (.md)",
        data=markdown,
        file_name=f"{context.lower().replace(' ', '_')}_brief.md",
        mime="text/markdown",
        key=f"download_{context}",
    )
    with st.expander("Numbers the AI was given"):
        st.json(numbers)
    st.markdown("**Copy as text**")
    st.code(markdown, language="markdown")


def generate_brief_button(context, numbers, status=None):
    audience = st.session_state.brief_audience
    if status is None:
        status = status_from_metrics(numbers)
    if st.button("Generate brief", type="primary", key=f"brief_{context}"):
        with st.spinner("Generating stakeholder brief from the computed numbers..."):
            try:
                brief = generate_stakeholder_brief(numbers, context, audience)
                st.session_state[f"brief_result_{context}"] = (brief, status, audience, numbers)
            except Exception as e:
                st.error(f"Could not generate the brief: {e}")
    stored = st.session_state.get(f"brief_result_{context}")
    if stored:
        brief, stored_status, stored_audience, stored_numbers = stored
        st.markdown("---")
        st.subheader("Stakeholder brief")
        render_brief(brief, stored_status, stored_audience, context, stored_numbers)


def csv_download(df, label, filename, key):
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("📊 Olist BI Platform")
st.sidebar.caption("SQL analytics + stakeholder-ready AI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Executive Overview",
        "👥 Customer Intelligence (RFM)",
        "📈 Cohort Retention",
        "📦 Product & Category",
        "🏪 Seller Performance",
        "🗺️ Geographic Analysis",
        "🔄 New vs Returning",
        "💬 Ask the Data (Groq)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Brief settings")
st.session_state.brief_audience = st.sidebar.selectbox(
    "Brief audience",
    ["Executive", "Marketing", "Operations", "Finance"],
    key="brief_audience_select",
)
st.session_state.min_monthly_orders = st.sidebar.slider(
    "Minimum monthly orders",
    min_value=0,
    max_value=500,
    value=DEFAULT_MIN_MONTHLY_ORDERS,
    step=10,
    help="Months below this order count are excluded from monthly comparisons and KPI trends.",
)

with st.sidebar.expander("🔌 Connection Status"):
    if st.button("Test Database Connection"):
        if test_connection():
            st.success("Connected to Supabase")
        else:
            st.error("Connection failed. Check DATABASE_URL.")

st.sidebar.info(
    f"Targets used by callouts: on-time ≥ {ON_TIME_TARGET:.0f}% · rating ≥ {GOOD_RATING:.1f}"
)


# ============================================================
# Page 1: Executive
# ============================================================
if page == "🏠 Executive Overview":
    st.title("Executive Overview")
    st.caption("Business health after excluding incomplete months below the selected order threshold.")
    try:
        kpis = filtered_kpis()
        rfm = load_rfm()
        if kpis.empty:
            st.warning("No months meet the current minimum-order filter. Lower the threshold in the sidebar.")
            st.stop()

        total_gmv = kpis["gmv"].sum()
        total_orders = kpis["total_orders"].sum()
        total_customers = rfm.shape[0]
        avg_aov = total_gmv / total_orders if total_orders else np.nan
        avg_rating = np.average(kpis["avg_review_score"], weights=kpis["total_orders"])
        avg_on_time = np.average(kpis["on_time_pct"], weights=kpis["total_orders"])
        prev = kpis.iloc[-2] if len(kpis) > 1 else None
        latest = kpis.iloc[-1]
        gmv_change = pct_change(latest["gmv"], prev["gmv"]) if prev is not None else np.nan

        cols = st.columns(6)
        cols[0].metric("Total GMV", f"R$ {total_gmv:,.0f}")
        cols[1].metric("Orders", f"{total_orders:,.0f}")
        cols[2].metric("Unique Customers", f"{total_customers:,.0f}")
        cols[3].metric("Weighted AOV", f"R$ {avg_aov:,.1f}")
        cols[4].metric("Weighted Rating", f"{avg_rating:.2f}")
        cols[5].metric("Weighted On-time", f"{avg_on_time:.1f}%")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=kpis["year_month"], y=kpis["gmv"], name="GMV"), secondary_y=False)
        fig.add_trace(go.Scatter(x=kpis["year_month"], y=kpis["total_orders"], name="Orders", mode="lines+markers"), secondary_y=True)
        fig.update_layout(title="Monthly GMV & Orders", height=420, hovermode="x unified")
        plot_chart(fig)

        c1, c2 = st.columns(2)
        with c1:
            plot_chart(px.line(kpis, x="year_month", y="aov", markers=True, title="Order Value Trend"))
        with c2:
            plot_chart(px.line(kpis, x="year_month", y="on_time_pct", markers=True, title="On-time Delivery Trend"))

        st.subheader("What stands out")
        if gmv_change == gmv_change:
            direction = "increased" if gmv_change >= 0 else "decreased"
            st.info(f"Latest eligible month ({latest['year_month']}) GMV {direction} {abs(gmv_change):.1f}% versus the previous eligible month. Weighted on-time delivery is {avg_on_time:.1f}%.")
        else:
            st.info(f"Weighted on-time delivery is {avg_on_time:.1f}%; there is not enough eligible history for a month-over-month comparison.")

        numbers = {
            "eligible_months": int(len(kpis)),
            "minimum_monthly_orders_filter": int(st.session_state.min_monthly_orders),
            "total_gmv": round(float(total_gmv), 2),
            "total_orders": int(total_orders),
            "unique_customers": int(total_customers),
            "weighted_average_order_value": round(float(avg_aov), 2),
            "weighted_average_review_score": round(float(avg_rating), 3),
            "weighted_on_time_pct": round(float(avg_on_time), 3),
            "latest_eligible_month": str(latest["year_month"]),
            "latest_vs_previous_gmv_pct": None if gmv_change != gmv_change else round(float(gmv_change), 2),
        }
        generate_brief_button("Executive Overview", numbers, status_from_metrics({
            "on_time_pct": avg_on_time, "avg_rating": avg_rating
        }))
    except Exception as e:
        st.error(f"Error loading data: {e}")


# ============================================================
# Page 2: RFM
# ============================================================
elif page == "👥 Customer Intelligence (RFM)":
    st.title("Customer Intelligence — RFM Segmentation")
    st.caption("Recency, frequency and monetary value for customer prioritisation.")
    try:
        rfm = load_rfm()
        seg = rfm.groupby("rfm_segment").agg(
            customers=("customer_unique_id", "count"),
            total_revenue=("total_revenue", "sum"),
            avg_revenue=("total_revenue", "mean"),
            avg_orders=("total_orders", "mean"),
        ).reset_index().sort_values("total_revenue", ascending=False)
        total_revenue = seg["total_revenue"].sum()
        seg["revenue_share_pct"] = seg["total_revenue"] / total_revenue * 100

        c1, c2 = st.columns([1.2, 1])
        with c1:
            plot_chart(px.bar(seg.sort_values("customers", ascending=False), x="rfm_segment", y="customers", color="rfm_segment", title="Customers by RFM Segment"))
        with c2:
            display = seg.copy()
            show_df(display.style.format({
                "total_revenue": "R$ {:,.0f}", "avg_revenue": "R$ {:,.1f}",
                "avg_orders": "{:.2f}", "revenue_share_pct": "{:.1f}%"
            }), height=400)

        plot_chart(px.scatter(
            rfm.sample(min(3000, len(rfm)), random_state=42),
            x="recency_days", y="total_revenue", color="f_score",
            size="total_orders", hover_data=["rfm_segment"], title="Customer Value Landscape"
        ))

        csv_download(seg, "Download segment table", "rfm_segments.csv", "csv_rfm")

        top_share = seg[seg["rfm_segment"].isin(["Champions", "Loyal Customers"])]["revenue_share_pct"].sum()
        numbers = {
            "customers_analyzed": int(len(rfm)),
            "segment_customer_counts": {str(k): int(v) for k, v in rfm["rfm_segment"].value_counts().items()},
            "segment_revenue": {str(r["rfm_segment"]): round(float(r["total_revenue"]), 2) for _, r in seg.iterrows()},
            "segment_revenue_share_pct": {str(r["rfm_segment"]): round(float(r["revenue_share_pct"]), 2) for _, r in seg.iterrows()},
            "champions_plus_loyal_revenue_share_pct": round(float(top_share), 2),
        }
        generate_brief_button("RFM Customer Segmentation", numbers)
    except Exception as e:
        st.error(f"Error: {e}")


# ============================================================
# Page 3: Cohort
# ============================================================
elif page == "📈 Cohort Retention":
    st.title("Cohort Retention Analysis")
    st.caption("Retention is shown by acquisition cohort; small/incomplete months are not silently used for monthly KPI comparisons.")
    try:
        cohort = load_cohort()
        pivot = cohort.pivot(index="cohort_month", columns="month_number", values="retention_rate").sort_index()
        plot_chart(px.imshow(
            pivot, labels={"x": "Months Since First Purchase", "y": "Cohort Month", "color": "Retention %"},
            color_continuous_scale="Blues", aspect="auto", title="Customer Retention Heatmap (%)"
        ))

        selected = st.multiselect("Select cohorts to compare", sorted(cohort["cohort_month"].unique()), default=sorted(cohort["cohort_month"].unique())[:6])
        if selected:
            plot_chart(px.line(cohort[cohort["cohort_month"].isin(selected)], x="month_number", y="retention_rate", color="cohort_month", markers=True, title="Retention Curves"))

        m1 = cohort.loc[cohort["month_number"] == 1, "retention_rate"].mean()
        m3 = cohort.loc[cohort["month_number"] == 3, "retention_rate"].mean()
        csv_download(cohort, "Download cohort data", "cohort_retention.csv", "csv_cohort")

        numbers = {
            "month_1_average_retention_pct": None if pd.isna(m1) else round(float(m1), 2),
            "month_3_average_retention_pct": None if pd.isna(m3) else round(float(m3), 2),
            "cohorts_analyzed": int(cohort["cohort_month"].nunique()),
        }
        generate_brief_button("Cohort Retention", numbers, status_from_metrics({"m1_retention": m1}))
    except Exception as e:
        st.error(f"Error: {e}")


# ============================================================
# Page 4: Product
# ============================================================
elif page == "📦 Product & Category":
    st.title("Product & Category Performance")
    try:
        prod = load_product()
        plot_chart(px.bar(prod.head(15), x="revenue", y="category", orientation="h", title="Top 15 Categories by Revenue", color="revenue"))
        plot_chart(px.scatter(prod, x="orders", y="avg_rating", size="revenue", color="avg_price", hover_name="category", title="Orders vs Rating"))
        show_df(prod.style.format({"revenue": "R$ {:,.0f}", "avg_price": "R$ {:,.1f}", "avg_rating": "{:.2f}"}))
        csv_download(prod, "Download product table", "product_performance.csv", "csv_product")

        total_revenue = prod["revenue"].sum()
        top5 = prod.head(5)
        numbers = {
            "categories_in_table": int(len(prod)),
            "top_category": str(prod.iloc[0]["category"]),
            "top_category_revenue": round(float(prod.iloc[0]["revenue"]), 2),
            "top_5_revenue_share_of_loaded_table_pct": round(float(top5["revenue"].sum() / total_revenue * 100), 2) if total_revenue else 0,
            "top_5_average_rating": round(float(top5["avg_rating"].mean()), 3),
        }
        generate_brief_button("Product & Category Performance", numbers)
    except Exception as e:
        st.error(f"Error: {e}")


# ============================================================
# Page 5: Seller
# ============================================================
elif page == "🏪 Seller Performance":
    st.title("Seller Performance & Delivery Quality")
    try:
        sellers = load_seller()
        sellers["needs_review"] = (sellers["on_time_pct"] < ON_TIME_TARGET) | (sellers["avg_rating"] < GOOD_RATING)

        plot_chart(px.bar(sellers.head(15), x="revenue", y="seller_id", orientation="h", title="Top Sellers by Revenue", color="avg_rating"))
        plot_chart(px.scatter(
            sellers, x="avg_delivery_days", y="avg_rating", size="revenue", color="on_time_pct",
            hover_data=["seller_id", "orders_fulfilled"], title="Delivery Speed vs Rating"
        ))

        review = sellers[sellers["needs_review"]].sort_values("revenue", ascending=False).head(20)
        st.subheader("Sellers to review first")
        show_df(review)
        csv_download(sellers, "Download seller table", "seller_performance.csv", "csv_seller")

        numbers = {
            "sellers_loaded": int(len(sellers)),
            "review_threshold_on_time_pct": ON_TIME_TARGET,
            "review_threshold_rating": GOOD_RATING,
            "sellers_flagged": int(sellers["needs_review"].sum()),
            "flagged_seller_revenue": round(float(review["revenue"].sum()), 2),
            "top_seller_revenue": round(float(sellers.iloc[0]["revenue"]), 2),
            "top_seller_on_time_pct": round(float(sellers.iloc[0]["on_time_pct"]), 2),
            "top_seller_rating": round(float(sellers.iloc[0]["avg_rating"]), 2),
        }
        generate_brief_button("Seller Performance", numbers)
    except Exception as e:
        st.error(f"Error: {e}")


# ============================================================
# Page 6: Geography
# ============================================================
elif page == "🗺️ Geographic Analysis":
    st.title("Geographic Performance by State")
    try:
        state = load_state()
        plot_chart(px.bar(state, x="customer_state", y="gmv", color="on_time_pct", title="GMV by State"))
        plot_chart(px.scatter(state, x="avg_delivery_days", y="avg_rating", size="gmv", color="customer_state", title="Delivery Days vs Rating by State"))
        show_df(state.style.format({"gmv": "R$ {:,.0f}", "aov": "R$ {:,.1f}", "avg_delivery_days": "{:.1f}", "on_time_pct": "{:.1f}%"}))
        csv_download(state, "Download state table", "state_performance.csv", "csv_state")

        worst_delivery = state.sort_values("on_time_pct").iloc[0]
        numbers = {
            "states_analyzed": int(len(state)),
            "highest_gmv_state": str(state.sort_values("gmv", ascending=False).iloc[0]["customer_state"]),
            "highest_gmv": round(float(state["gmv"].max()), 2),
            "lowest_on_time_state": str(worst_delivery["customer_state"]),
            "lowest_on_time_pct": round(float(worst_delivery["on_time_pct"]), 2),
            "overall_state_gmv": round(float(state["gmv"].sum()), 2),
        }
        generate_brief_button("Geographic Performance", numbers)
    except Exception as e:
        st.error(f"Error: {e}")


# ============================================================
# Page 7: New vs Returning
# ============================================================
elif page == "🔄 New vs Returning":
    st.title("New vs Returning Customers")
    try:
        nr = load_new_returning()
        plot_chart(px.bar(nr, x="purchase_month", y="revenue", color="customer_type", title="Revenue: New vs Returning", barmode="stack"))
        plot_chart(px.line(nr, x="purchase_month", y="customers", color="customer_type", markers=True, title="Customers Over Time"))

        pivot = nr.pivot(index="purchase_month", columns="customer_type", values="revenue").fillna(0)
        if {"Returning", "New"}.issubset(pivot.columns):
            pivot["returning_share"] = pivot["Returning"] / (pivot["Returning"] + pivot["New"]) * 100
            plot_chart(px.line(pivot, y="returning_share", markers=True, title="Returning Revenue Share"))

        show_df(nr)
        csv_download(nr, "Download new vs returning data", "new_vs_returning.csv", "csv_nr")

        total = nr["revenue"].sum()
        returning = nr.loc[nr["customer_type"] == "Returning", "revenue"].sum()
        returning_share = returning / total * 100 if total else np.nan
        numbers = {
            "total_revenue": round(float(total), 2),
            "new_revenue": round(float(nr.loc[nr["customer_type"] == "New", "revenue"].sum()), 2),
            "returning_revenue": round(float(returning), 2),
            "returning_revenue_share_pct": None if pd.isna(returning_share) else round(float(returning_share), 2),
        }
        generate_brief_button("New vs Returning Customers", numbers, status_from_metrics({"returning_revenue_share": returning_share}))
    except Exception as e:
        st.error(f"Error: {e}")


# ============================================================
# Page 8: Ask the Data
# ============================================================
else:
    st.title("💬 Ask the Data")
    st.caption("Legacy free-form explanations remain available. For page-specific memos, use Generate brief.")
    topic = st.selectbox("Choose a topic", [
        "Overall business health", "RFM segmentation insights", "Retention problem",
        "Delivery performance", "Top product categories", "Custom (paste your own numbers)"
    ])
    custom_text = ""
    if topic.startswith("Custom"):
        custom_text = st.text_area("Paste metrics or findings here", height=160)

    brief_numbers = None
    brief_context = f"Ask the Data — {topic}"
    if st.button("Generate Stakeholder Explanation", type="primary"):
        with st.spinner("Generating..."):
            if topic == "Overall business health":
                kpis = filtered_kpis()
                rfm = load_rfm()
                weighted_aov = kpis.gmv.sum() / kpis.total_orders.sum()
                weighted_on_time = np.average(kpis.on_time_pct, weights=kpis.total_orders)
                text = f"GMV: R$ {kpis.gmv.sum():,.0f}; Orders: {kpis.total_orders.sum():,.0f}; Customers: {len(rfm):,.0f}; Weighted AOV: R$ {weighted_aov:,.1f}; Weighted on-time: {weighted_on_time:.1f}%."
                result = explain_to_stakeholder(text, "Overall Olist Business Health")
                brief_numbers = {"gmv": round(float(kpis.gmv.sum()),2), "orders": int(kpis.total_orders.sum()), "customers": int(len(rfm)), "weighted_aov": round(float(weighted_aov),2), "weighted_on_time_pct": round(float(weighted_on_time),2)}
            elif topic == "RFM segmentation insights":
                rfm = load_rfm()
                seg = rfm.groupby("rfm_segment")["total_revenue"].sum().sort_values(ascending=False)
                result = explain_to_stakeholder(f"Segment revenue: {seg.to_dict()}", "RFM Customer Segmentation")
                brief_numbers = {"segment_revenue": {str(k): round(float(v),2) for k,v in seg.items()}}
            elif topic == "Retention problem":
                cohort = load_cohort()
                m1 = cohort.loc[cohort.month_number == 1, "retention_rate"].mean()
                m3 = cohort.loc[cohort.month_number == 3, "retention_rate"].mean()
                result = explain_to_stakeholder(f"Month-1 retention: {m1:.1f}%; Month-3 retention: {m3:.1f}%.", "Customer Cohort Retention")
                brief_numbers = {"month_1_retention_pct": round(float(m1),2), "month_3_retention_pct": round(float(m3),2)}
            elif topic == "Delivery performance":
                kpis = filtered_kpis()
                weighted_on_time = np.average(kpis.on_time_pct, weights=kpis.total_orders)
                weighted_days = np.average(kpis.avg_delivery_days, weights=kpis.total_orders)
                result = explain_to_stakeholder(f"Weighted on-time: {weighted_on_time:.1f}%; weighted average delivery days: {weighted_days:.1f}.", "Delivery Performance")
                brief_numbers = {"weighted_on_time_pct": round(float(weighted_on_time),2), "weighted_delivery_days": round(float(weighted_days),2)}
            elif topic == "Top product categories":
                prod = load_product()
                result = explain_to_stakeholder(str(prod.head(5)[["category", "revenue", "avg_rating"]].to_dict("records")), "Product Performance")
                brief_numbers = {"top_5_categories": prod.head(5)[["category","revenue","avg_rating"]].to_dict("records")}
            else:
                result = explain_to_stakeholder(custom_text, "Custom Analysis")
                brief_numbers = {"user_supplied_metrics": custom_text}
        st.markdown("### Explanation")
        st.info(result)

    # Every page ends with the same auditable stakeholder brief workflow.
    if brief_numbers is not None:
        generate_brief_button(brief_context, brief_numbers)

st.sidebar.markdown("---")
st.sidebar.caption("Built for an analytics portfolio • SQL + Streamlit + Groq")
