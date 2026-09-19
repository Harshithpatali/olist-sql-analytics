import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from utils.db import run_query, test_connection
from utils.llm import explain_to_stakeholder

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Olist Business Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4e79a7;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    /* Better looking explanation boxes */
    div[data-testid="stAlert"] {
        padding: 1.2rem 1.4rem;
        border-radius: 10px;
        font-size: 1.02rem;
        line-height: 1.55;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Helper: Load data with caching
# --------------------------------------------------
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
    return run_query("SELECT * FROM vw_product_performance LIMIT 30")

@st.cache_data(ttl=3600)
def load_seller():
    return run_query("SELECT * FROM vw_seller_performance LIMIT 50")

@st.cache_data(ttl=3600)
def load_state():
    return run_query("SELECT * FROM vw_state_performance")

@st.cache_data(ttl=3600)
def load_new_returning():
    return run_query("SELECT * FROM vw_new_vs_returning ORDER BY purchase_month")

@st.cache_data(ttl=3600)
def load_customer_metrics():
    return run_query("SELECT * FROM vw_customer_metrics")


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("📊 Olist BI Platform")
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
        "💬 Ask the Data (Groq)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard uses real analytical views that Data Scientists build in companies:\n"
    "- RFM Segmentation\n"
    "- Cohort Retention\n"
    "- Business KPIs\n"
    "- Delivery SLAs\n"
    "- Product & Seller performance"
)

# Connection check
with st.sidebar.expander("🔌 Connection Status"):
    if st.button("Test Database Connection"):
        if test_connection():
            st.success("✅ Connected to Supabase")
        else:
            st.error("❌ Connection failed. Check DATABASE_URL")


# --------------------------------------------------
# PAGE 1: Executive Overview
# --------------------------------------------------
if page == "🏠 Executive Overview":
    st.title("Executive Overview")
    st.caption("High-level business health metrics from the Olist marketplace")

    try:
        kpis = load_monthly_kpis()
        rfm = load_rfm()

        # Top metrics
        total_gmv = kpis["gmv"].sum()
        total_orders = kpis["total_orders"].sum()
        total_customers = rfm.shape[0]
        avg_aov = kpis["aov"].mean()
        avg_rating = kpis["avg_review_score"].mean()
        avg_on_time = kpis["on_time_pct"].mean()

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total GMV (R$)", f"{total_gmv:,.0f}")
        col2.metric("Total Orders", f"{total_orders:,.0f}")
        col3.metric("Unique Customers", f"{total_customers:,.0f}")
        col4.metric("Avg Order Value", f"R$ {avg_aov:,.1f}")
        col5.metric("Avg Review Score", f"{avg_rating:.2f}")
        col6.metric("On-Time Delivery %", f"{avg_on_time:.1f}%")

        st.markdown("---")

        # GMV & Orders trend
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=kpis["year_month"], y=kpis["gmv"], name="GMV (R$)", marker_color="#4e79a7"),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=kpis["year_month"], y=kpis["total_orders"], name="Orders", line=dict(color="#f28e2b", width=3)),
            secondary_y=True
        )
        fig.update_layout(title="Monthly GMV & Order Volume", height=420, hovermode="x unified")
        fig.update_yaxes(title_text="GMV (R$)", secondary_y=False)
        fig.update_yaxes(title_text="Orders", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)

        with col_a:
            fig2 = px.line(kpis, x="year_month", y="aov", markers=True, title="Average Order Value Trend")
            fig2.update_traces(line_color="#59a14f")
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            fig3 = px.line(kpis, x="year_month", y="on_time_pct", markers=True, title="On-Time Delivery %")
            fig3.update_traces(line_color="#e15759")
            st.plotly_chart(fig3, use_container_width=True)

        # Quick insight box
        st.markdown("### 📌 Key Takeaways")
        latest = kpis.iloc[-1]
        prev = kpis.iloc[-2] if len(kpis) > 1 else latest
        gmv_change = ((latest["gmv"] - prev["gmv"]) / prev["gmv"] * 100) if prev["gmv"] != 0 else 0

        st.info(f"""
        - Latest month (**{latest['year_month']}**) GMV: **R$ {latest['gmv']:,.0f}** ({gmv_change:+.1f}% vs previous month)
        - Average delivery time across the period: **{kpis['avg_delivery_days'].mean():.1f} days**
        - Overall on-time delivery rate: **{avg_on_time:.1f}%**
        """)

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure you have run the SQL views in Supabase first.")


# --------------------------------------------------
# PAGE 2: Customer Intelligence (RFM)
# --------------------------------------------------
elif page == "👥 Customer Intelligence (RFM)":
    st.title("Customer Intelligence – RFM Segmentation")
    st.caption("Classic Recency-Frequency-Monetary analysis used by virtually every growth / CRM team")

    try:
        rfm = load_rfm()

        # Segment distribution
        seg_counts = rfm["rfm_segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Customers"]

        col1, col2 = st.columns([1.2, 1])

        with col1:
            fig = px.bar(
                seg_counts, x="Segment", y="Customers", color="Segment",
                title="Customer Distribution by RFM Segment",
                text_auto=True
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            seg_value = rfm.groupby("rfm_segment").agg(
                customers=("customer_unique_id", "count"),
                total_revenue=("total_revenue", "sum"),
                avg_revenue=("total_revenue", "mean"),
                avg_orders=("total_orders", "mean")
            ).reset_index().sort_values("total_revenue", ascending=False)

            st.dataframe(
                seg_value.style.format({
                    "total_revenue": "R$ {:,.0f}",
                    "avg_revenue": "R$ {:,.1f}",
                    "avg_orders": "{:.2f}"
                }),
                use_container_width=True,
                height=400
            )

        st.markdown("---")
        st.subheader("RFM Score Distributions")

        c1, c2, c3 = st.columns(3)
        with c1:
            fig_r = px.histogram(rfm, x="r_score", nbins=5, title="Recency Score", color_discrete_sequence=["#4e79a7"])
            st.plotly_chart(fig_r, use_container_width=True)
        with c2:
            fig_f = px.histogram(rfm, x="f_score", nbins=5, title="Frequency Score", color_discrete_sequence=["#f28e2b"])
            st.plotly_chart(fig_f, use_container_width=True)
        with c3:
            fig_m = px.histogram(rfm, x="m_score", nbins=5, title="Monetary Score", color_discrete_sequence=["#59a14f"])
            st.plotly_chart(fig_m, use_container_width=True)

        # Scatter
        st.subheader("Recency vs Monetary (colored by Frequency)")
        fig_scatter = px.scatter(
            rfm.sample(min(3000, len(rfm))),  # sample for performance
            x="recency_days", y="total_revenue",
            color="f_score", size="total_orders",
            hover_data=["rfm_segment"],
            title="Customer Value Landscape",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Explain button
        if st.button("🧠 Explain RFM findings to a stakeholder"):
            summary = f"""
            Total customers analyzed: {len(rfm)}
            Segment distribution: {dict(seg_counts.values)}
            Champions & Loyal customers contribute the majority of revenue.
            Average revenue of Champions vs Lost customers shows clear value difference.
            """
            with st.spinner("Generating explanation..."):
                explanation = explain_to_stakeholder(summary, context="RFM Customer Segmentation Analysis")
            st.markdown("### 📝 Stakeholder-Ready Explanation")
            st.info(explanation)

    except Exception as e:
        st.error(f"Error: {e}")


# --------------------------------------------------
# PAGE 3: Cohort Retention
# --------------------------------------------------
elif page == "📈 Cohort Retention":
    st.title("Cohort Retention Analysis")
    st.caption("How well do we retain customers over time? (Standard growth metric)")

    try:
        cohort = load_cohort()

        # Pivot for heatmap
        pivot = cohort.pivot(index="cohort_month", columns="month_number", values="retention_rate")
        pivot = pivot.sort_index()

        fig = px.imshow(
            pivot,
            labels=dict(x="Months Since First Purchase", y="Cohort Month", color="Retention %"),
            color_continuous_scale="Blues",
            aspect="auto",
            title="Customer Retention Heatmap (%)"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Retention Curves by Cohort")
        # Line chart for selected cohorts
        selected_cohorts = st.multiselect(
            "Select cohorts to compare",
            options=sorted(cohort["cohort_month"].unique()),
            default=sorted(cohort["cohort_month"].unique())[:6]
        )

        if selected_cohorts:
            filtered = cohort[cohort["cohort_month"].isin(selected_cohorts)]
            fig2 = px.line(
                filtered, x="month_number", y="retention_rate",
                color="cohort_month", markers=True,
                title="Retention Rate Over Time"
            )
            fig2.update_layout(yaxis_title="Retention %", xaxis_title="Months Since Acquisition")
            st.plotly_chart(fig2, use_container_width=True)

        if st.button("🧠 Explain cohort retention to a stakeholder"):
            avg_m1 = cohort[cohort["month_number"] == 1]["retention_rate"].mean()
            avg_m3 = cohort[cohort["month_number"] == 3]["retention_rate"].mean()
            summary = f"""
            Month-1 average retention: {avg_m1:.1f}%
            Month-3 average retention: {avg_m3:.1f}%
            Retention drops significantly after the first month, which is common in marketplaces.
            """
            with st.spinner("Generating explanation..."):
                explanation = explain_to_stakeholder(summary, context="Cohort Retention Analysis")
            st.markdown("### 📝 Stakeholder-Ready Explanation")
            st.info(explanation)

    except Exception as e:
        st.error(f"Error: {e}")


# --------------------------------------------------
# PAGE 4: Product & Category
# --------------------------------------------------
elif page == "📦 Product & Category":
    st.title("Product & Category Performance")

    try:
        prod = load_product()

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                prod.head(15), x="revenue", y="category",
                orientation="h", title="Top 15 Categories by Revenue",
                color="revenue", color_continuous_scale="Teal"
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(
                prod, x="orders", y="avg_rating", size="revenue",
                color="avg_price", hover_name="category",
                title="Orders vs Rating (size = Revenue)",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(
            prod.style.format({
                "revenue": "R$ {:,.0f}",
                "avg_price": "R$ {:,.1f}",
                "avg_rating": "{:.2f}"
            }),
            use_container_width=True
        )

        if st.button("🧠 Explain product findings"):
            top_cat = prod.iloc[0]
            summary = f"""
            Top category by revenue: {top_cat['category']} with R$ {top_cat['revenue']:,.0f}
            Average rating of top categories is around {prod.head(10)['avg_rating'].mean():.2f}
            There is variation in price points and volume across categories.
            """
            with st.spinner("Generating..."):
                st.markdown("### 📝 Stakeholder-Ready Explanation")
                st.info(explain_to_stakeholder(summary, "Product Category Performance"))

    except Exception as e:
        st.error(f"Error: {e}")


# --------------------------------------------------
# PAGE 5: Seller Performance
# --------------------------------------------------
elif page == "🏪 Seller Performance":
    st.title("Seller Performance & Delivery Quality")

    try:
        sellers = load_seller()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                sellers.head(15), x="revenue", y="seller_id",
                orientation="h", title="Top 15 Sellers by Revenue",
                color="avg_rating", color_continuous_scale="RdYlGn"
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(
                sellers, x="avg_delivery_days", y="avg_rating",
                size="revenue", color="on_time_pct",
                title="Delivery Speed vs Rating (color = On-time %)",
                color_continuous_scale="RdYlGn",
                hover_data=["seller_id", "orders_fulfilled"]
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(sellers.head(20), use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")


# --------------------------------------------------
# PAGE 6: Geographic
# --------------------------------------------------
elif page == "🗺️ Geographic Analysis":
    st.title("Geographic Performance by State")

    try:
        state = load_state()

        fig = px.bar(
            state, x="customer_state", y="gmv",
            color="on_time_pct", color_continuous_scale="RdYlGn",
            title="GMV by State (color = On-time Delivery %)",
            text_auto=".2s"
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.scatter(
                state, x="avg_delivery_days", y="avg_rating",
                size="gmv", color="customer_state",
                title="Delivery Days vs Customer Rating by State"
            )
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            st.dataframe(
                state.style.format({
                    "gmv": "R$ {:,.0f}",
                    "aov": "R$ {:,.1f}",
                    "avg_delivery_days": "{:.1f}",
                    "on_time_pct": "{:.1f}%"
                }),
                use_container_width=True,
                height=400
            )

    except Exception as e:
        st.error(f"Error: {e}")


# --------------------------------------------------
# PAGE 7: New vs Returning
# --------------------------------------------------
elif page == "🔄 New vs Returning":
    st.title("New vs Returning Customers")

    try:
        nr = load_new_returning()

        fig = px.bar(
            nr, x="purchase_month", y="revenue", color="customer_type",
            title="Revenue Contribution: New vs Returning Customers",
            barmode="stack",
            color_discrete_map={"New": "#4e79a7", "Returning": "#f28e2b"}
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(
            nr, x="purchase_month", y="customers", color="customer_type",
            markers=True, title="Number of New vs Returning Customers Over Time"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Share calculation
        pivot_rev = nr.pivot(index="purchase_month", columns="customer_type", values="revenue").fillna(0)
        if "Returning" in pivot_rev.columns and "New" in pivot_rev.columns:
            pivot_rev["returning_share"] = pivot_rev["Returning"] / (pivot_rev["New"] + pivot_rev["Returning"]) * 100
            fig3 = px.line(pivot_rev, y="returning_share", markers=True,
                           title="% of Revenue from Returning Customers")
            st.plotly_chart(fig3, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")


# --------------------------------------------------
# PAGE 8: Ask the Data (Groq)
# --------------------------------------------------
elif page == "💬 Ask the Data (Groq)":
    st.title("💬 Ask the Data – Stakeholder Explanations")
    st.caption("Powered by Groq (Llama 3.3). Translate any finding into clear business language.")

    st.markdown("""
    **How to use:**
    1. Choose a topic or paste your own metrics.
    2. Click Generate.
    3. Copy the explanation into a slide or email.
    """)

    topic = st.selectbox(
        "Choose a pre-built topic",
        [
            "Overall business health",
            "RFM segmentation insights",
            "Retention problem",
            "Delivery performance",
            "Top product categories",
            "Custom (paste your own numbers)"
        ]
    )

    custom_text = ""
    if topic == "Custom (paste your own numbers)":
        custom_text = st.text_area("Paste metrics or findings here", height=150)

    if st.button("🚀 Generate Stakeholder Explanation", type="primary"):
        with st.spinner("Thinking like a senior analyst..."):
            if topic == "Overall business health":
                kpis = load_monthly_kpis()
                rfm = load_rfm()
                text = f"""
                Total GMV: R$ {kpis['gmv'].sum():,.0f}
                Total Orders: {kpis['total_orders'].sum():,.0f}
                Unique Customers: {len(rfm)}
                Average Order Value: R$ {kpis['aov'].mean():.1f}
                Average Review Score: {kpis['avg_review_score'].mean():.2f}
                On-time Delivery: {kpis['on_time_pct'].mean():.1f}%
                """
                result = explain_to_stakeholder(text, "Overall Olist Business Health")
            elif topic == "RFM segmentation insights":
                rfm = load_rfm()
                seg = rfm["rfm_segment"].value_counts().to_dict()
                text = f"RFM Segment counts: {seg}. Champions and Loyal customers drive most revenue."
                result = explain_to_stakeholder(text, "RFM Customer Segmentation")
            elif topic == "Retention problem":
                cohort = load_cohort()
                m1 = cohort[cohort["month_number"] == 1]["retention_rate"].mean()
                m3 = cohort[cohort["month_number"] == 3]["retention_rate"].mean()
                text = f"Month-1 retention ≈ {m1:.1f}%. Month-3 retention ≈ {m3:.1f}%."
                result = explain_to_stakeholder(text, "Customer Cohort Retention")
            elif topic == "Delivery performance":
                kpis = load_monthly_kpis()
                text = f"Average on-time delivery: {kpis['on_time_pct'].mean():.1f}%. Avg delivery days: {kpis['avg_delivery_days'].mean():.1f}."
                result = explain_to_stakeholder(text, "Delivery & Logistics Performance")
            elif topic == "Top product categories":
                prod = load_product()
                top = prod.head(5)[["category", "revenue", "avg_rating"]].to_dict("records")
                text = f"Top 5 categories: {top}"
                result = explain_to_stakeholder(text, "Product Category Performance")
            else:
                result = explain_to_stakeholder(custom_text, "Custom Analysis")

            st.markdown("### 📝 Stakeholder-Ready Explanation")
            st.info(result)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Built for entry-level Data Science portfolio • SQL + Streamlit + Groq")
