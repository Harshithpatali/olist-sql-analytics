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
.block-container {
    padding-top: 1.25rem;
    padding-bottom: 2rem;
}

/* Dark-theme KPI cards: avoid the white blocks visible in dark mode. */
[data-testid="stMetric"] {
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 14px;
    padding: 12px 14px;
    background: rgba(30, 41, 59, 0.72);
    box-shadow: none;
}

[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #f8fafc !important;
}

[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #cbd5e1 !important;
}

.brief-card {
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 14px;
    padding: 18px;
    background: rgba(30, 41, 59, 0.72);
    margin: 8px 0 18px 0;
}

.status-pill {
    display: inline-block;
    padding: 5px 11px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.82rem;
    margin-bottom: 8px;
}

.status-strong { background:#143d2a; color:#86efac; }
.status-stable { background:#172f52; color:#93c5fd; }
.status-watch { background:#4a3510; color:#fcd34d; }
.status-risk { background:#4a1d1d; color:#fca5a5; }

.finding-number {
    font-size: 1.65rem;
    font-weight: 750;
    line-height: 1.1;
    color: #f8fafc;
}

.finding-label {
    font-weight: 650;
    margin-top: 3px;
    color: #e2e8f0;
}

.muted {
    color: #cbd5e1;
    font-size: .9rem;
}

div[data-testid="stExpander"] {
    border-color: rgba(148, 163, 184, 0.22);
}

div[data-testid="stDownloadButton"] button {
    width: 100%;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# Data access + cached transformations
# ============================================================

def _normalize_percent_column(df, column):
    """Normalize percentage columns that may be stored as 0-1 or 0-100."""
    df = df.copy()
    if column not in df.columns or df.empty:
        return df

    values = pd.to_numeric(df[column], errors="coerce")
    finite = values.dropna()

    if not finite.empty and finite.max() <= 1.000001:
        values = values * 100.0

    df[column] = values
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_monthly_kpis():
    df = run_query(
        "SELECT * FROM vw_monthly_kpis ORDER BY purchase_month"
    )
    df["total_orders"] = pd.to_numeric(
        df["total_orders"], errors="coerce"
    )
    df["gmv"] = pd.to_numeric(
        df["gmv"], errors="coerce"
    )
    df["avg_review_score"] = pd.to_numeric(
        df["avg_review_score"], errors="coerce"
    )
    df["on_time_pct"] = pd.to_numeric(
        df["on_time_pct"], errors="coerce"
    )
    return _normalize_percent_column(df, "on_time_pct")


@st.cache_data(ttl=3600, show_spinner=False)
def load_rfm():
    return run_query("SELECT * FROM vw_rfm_segments")


@st.cache_data(ttl=3600, show_spinner=False)
def load_cohort():
    return run_query("SELECT * FROM vw_cohort_retention")


@st.cache_data(ttl=3600, show_spinner=False)
def load_product():
    return run_query(
        "SELECT * FROM vw_product_performance "
        "ORDER BY revenue DESC LIMIT 30"
    )


@st.cache_data(ttl=3600, show_spinner=False)
def load_seller():
    df = run_query(
        "SELECT * FROM vw_seller_performance "
        "ORDER BY revenue DESC LIMIT 50"
    )
    return _normalize_percent_column(df, "on_time_pct")


@st.cache_data(ttl=3600, show_spinner=False)
def load_state():
    df = run_query(
        "SELECT * FROM vw_state_performance"
    )
    return _normalize_percent_column(df, "on_time_pct")


@st.cache_data(ttl=3600, show_spinner=False)
def load_new_returning():
    return run_query(
        "SELECT * FROM vw_new_vs_returning "
        "ORDER BY purchase_month"
    )


@st.cache_data(ttl=3600, show_spinner=False)
def load_customer_metrics():
    return run_query(
        "SELECT * FROM vw_customer_metrics"
    )


@st.cache_data(ttl=3600, show_spinner=False)
def get_filtered_kpis(min_monthly_orders):
    kpis = load_monthly_kpis().copy()
    kpis["total_orders"] = pd.to_numeric(
        kpis["total_orders"], errors="coerce"
    )
    return kpis[
        kpis["total_orders"] >= int(min_monthly_orders)
    ].copy()


@st.cache_data(ttl=3600, show_spinner=False)
def build_rfm_segments(rfm):
    seg = (
        rfm.groupby("rfm_segment")
        .agg(
            customers=("customer_unique_id", "count"),
            total_revenue=("total_revenue", "sum"),
            avg_revenue=("total_revenue", "mean"),
            avg_orders=("total_orders", "mean"),
        )
        .reset_index()
        .sort_values(
            "total_revenue",
            ascending=False,
        )
    )

    total_revenue = seg["total_revenue"].sum()

    if total_revenue:
        seg["revenue_share_pct"] = (
            seg["total_revenue"]
            / total_revenue
            * 100
        )
    else:
        seg["revenue_share_pct"] = 0.0

    return seg


@st.cache_data(ttl=3600, show_spinner=False)
def sample_rfm(rfm, sample_size=3000):
    return rfm.sample(
        min(sample_size, len(rfm)),
        random_state=42,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def build_seller_review(sellers, on_time_target, rating_target):
    result = sellers.copy()

    result["needs_review"] = (
        result["on_time_pct"] < on_time_target
    ) | (
        result["avg_rating"] < rating_target
    )

    review = (
        result[result["needs_review"]]
        .sort_values(
            "revenue",
            ascending=False,
        )
        .head(20)
        .copy()
    )

    return result, review


def filtered_kpis():
    return get_filtered_kpis(
        st.session_state.min_monthly_orders
    )


def plot_chart(fig, **kwargs):
    """Render charts consistently with the dark application theme."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=50, r=30, t=55, b=45),
    )

    try:
        st.plotly_chart(
            fig,
            width="stretch",
            **kwargs,
        )
    except TypeError:
        st.plotly_chart(
            fig,
            use_container_width=True,
            **kwargs,
        )


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

    on_time = numbers.get("on_time_pct")
    rating = numbers.get("avg_rating")
    returning_share = numbers.get(
        "returning_revenue_share",
        numbers.get("returning_share"),
    )
    m1_retention = numbers.get("m1_retention")

    if on_time is not None and not pd.isna(on_time):
        on_time_value = float(on_time)
        if on_time_value <= 1.0:
            on_time_value *= 100.0
        checks.append(on_time_value >= ON_TIME_TARGET)

    if rating is not None and not pd.isna(rating):
        checks.append(float(rating) >= GOOD_RATING)

    if returning_share is not None and not pd.isna(returning_share):
        checks.append(float(returning_share) >= 30)

    if m1_retention is not None and not pd.isna(m1_retention):
        checks.append(float(m1_retention) >= 20)

    if not checks:
        return "Stable"

    score = sum(bool(x) for x in checks) / len(checks)

    if score >= 0.80:
        return "Strong"
    if score >= 0.50:
        return "Stable"
    if score >= 0.25:
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


def render_brief(
    brief,
    status,
    audience,
    context,
    numbers,
):
    render_status(status)

    headline = brief.get("headline", "Stakeholder conclusion")
    st.markdown(f"### {headline}")

    summary = brief.get("summary", "")
    if summary:
        st.write(summary)

    st.markdown("### Key findings")

    findings = brief.get("findings", [])
    findings = findings[:4]

    if findings:
        cols = st.columns(len(findings))

        for col, item in zip(cols, findings):
            with col:
                number = str(item.get("number", ""))
                label = str(item.get("label", ""))
                detail = str(item.get("detail", ""))

                st.markdown(
                    f"""
                    <div class="brief-card">
                        <div class="finding-number">
                            {html.escape(number)}
                        </div>
                        <div class="finding-label">
                            {html.escape(label)}
                        </div>
                        <div class="muted">
                            {html.escape(detail)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("### Risks")

    risks = brief.get("risks", [])

    if risks:
        for risk in risks[:2]:
            st.markdown(f"- {risk}")
    else:
        st.caption(
            "No material risks were identified from the supplied metrics."
        )

    st.markdown("### Prioritised actions")

    actions = sorted(
        brief.get("actions", [])[:3],
        key=lambda x: x.get("priority", 99),
    )

    for action in actions:
        priority = action.get("priority", "")
        action_text = action.get("action", "")
        reason = action.get("reason", "")
        impact = action.get("expected_impact", "")

        st.markdown(f"**{priority}. {action_text}**")
        st.markdown(f"- **Reason:** {reason}")
        st.markdown(f"- **Expected impact:** {impact}")

    caveat = brief.get(
        "caveat",
        (
            "These figures describe the selected data and filters "
            "and do not establish causation."
        ),
    )

    st.caption(f"**How to read the numbers:** {caveat}")

    markdown = brief_to_markdown(
        brief=brief,
        status=status,
        audience=audience,
        context=context,
    )

    safe_filename = (
        context.lower()
        .replace(" ", "_")
        .replace("&", "and")
        .replace("/", "_")
    )

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "⬇️ Download Markdown",
            data=markdown,
            file_name=f"{safe_filename}_brief.md",
            mime="text/markdown",
            width="stretch",
            key=f"download_md_{safe_filename}",
        )

    with col2:
        st.download_button(
            "⬇️ Download Text",
            data=markdown,
            file_name=f"{safe_filename}_brief.txt",
            mime="text/plain",
            width="stretch",
            key=f"download_txt_{safe_filename}",
        )

    with st.expander("🔍 Numbers the AI was given"):
        st.json(numbers)

    with st.expander("📋 Copy as text"):
        st.text_area(
            "Stakeholder brief",
            value=markdown,
            height=420,
            label_visibility="collapsed",
        )


def generate_brief_button(context, numbers, status=None):
    audience = st.session_state.brief_audience

    if status is None:
        status = status_from_metrics(numbers)

    button_key = (
        "brief_"
        + context.lower()
        .replace(" ", "_")
        .replace("&", "and")
        .replace("/", "_")
        .replace("—", "_")
        .replace("–", "_")
    )

    if st.button(
        "Generate brief",
        type="primary",
        width="stretch",
        key=button_key,
    ):
        with st.spinner("Generating stakeholder brief..."):
            try:
                # Keep the payload small and JSON serializable.
                safe_numbers = json.loads(
                    json.dumps(numbers, default=str)
                )

                brief = generate_stakeholder_brief(
                    numbers=safe_numbers,
                    context=context,
                    audience=audience,
                )

                st.session_state[f"brief_result_{context}"] = {
                    "brief": brief,
                    "status": status,
                    "audience": audience,
                    "numbers": safe_numbers,
                }

                st.session_state[f"brief_error_{context}"] = None

            except Exception as e:
                st.session_state[f"brief_error_{context}"] = (
                    f"{type(e).__name__}: {e}"
                )

    error = st.session_state.get(f"brief_error_{context}")

    if error:
        st.error(
            f"Could not generate the brief: {error}"
        )

    stored = st.session_state.get(f"brief_result_{context}")

    if not stored:
        return

    brief = stored["brief"]
    stored_status = stored["status"]
    stored_audience = stored["audience"]
    stored_numbers = stored["numbers"]

    st.markdown("---")
    st.subheader("Stakeholder brief")

    render_brief(
        brief,
        stored_status,
        stored_audience,
        context,
        stored_numbers,
    )


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
if "brief_audience" not in st.session_state:
    st.session_state.brief_audience = "Executive"

if "min_monthly_orders" not in st.session_state:
    st.session_state.min_monthly_orders = DEFAULT_MIN_MONTHLY_ORDERS

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
    if st.button("Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

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
        valid_rating = kpis["avg_review_score"].notna()
        valid_on_time = kpis["on_time_pct"].notna()

        avg_rating = (
            np.average(
                kpis.loc[valid_rating, "avg_review_score"],
                weights=kpis.loc[valid_rating, "total_orders"],
            )
            if valid_rating.any()
            else np.nan
        )

        avg_on_time = (
            np.average(
                kpis.loc[valid_on_time, "on_time_pct"],
                weights=kpis.loc[valid_on_time, "total_orders"],
            )
            if valid_on_time.any()
            else np.nan
        )
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
        seg = build_rfm_segments(rfm)

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
            sample_rfm(rfm),
            x="recency_days", y="total_revenue", color="f_score",
            size="total_orders", hover_data=["rfm_segment"], title="Customer Value Landscape"
        ))

        csv_download(seg, "Download segment table", "rfm_segments.csv", "csv_rfm")

        numbers = {
            "customers_analyzed": int(len(rfm)),
            "segment_customer_counts": {
                str(k): int(v)
                for k, v in rfm["rfm_segment"].value_counts().to_dict().items()
            },
            "segment_revenue": {
                str(r["rfm_segment"]): round(float(r["total_revenue"]), 2)
                for _, r in seg.iterrows()
            },
            "segment_revenue_share_pct": {
                str(r["rfm_segment"]): round(float(r["revenue_share_pct"]), 2)
                for _, r in seg.iterrows()
            },
            "top_segment_by_revenue": (
                str(seg.iloc[0]["rfm_segment"])
                if not seg.empty
                else None
            ),
            "top_segment_revenue_share_pct": (
                round(float(seg.iloc[0]["revenue_share_pct"]), 2)
                if not seg.empty
                else None
            ),
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
            "categories_loaded": int(len(prod)),
            "top_category": (
                str(prod.iloc[0]["category"])
                if not prod.empty
                else None
            ),
            "top_category_revenue": (
                round(float(prod.iloc[0]["revenue"]), 2)
                if not prod.empty
                else None
            ),
            "top_5_revenue_share_of_loaded_categories_pct": (
                round(
                    float(top5["revenue"].sum() / total_revenue * 100),
                    2,
                )
                if total_revenue
                else 0
            ),
            "top_5_average_rating": (
                round(float(top5["avg_rating"].mean()), 2)
                if not top5.empty
                else None
            ),
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
        sellers, review = build_seller_review(
            sellers,
            ON_TIME_TARGET,
            GOOD_RATING,
        )

        plot_chart(px.bar(sellers.head(15), x="revenue", y="seller_id", orientation="h", title="Top Sellers by Revenue", color="avg_rating"))
        plot_chart(px.scatter(
            sellers, x="avg_delivery_days", y="avg_rating", size="revenue", color="on_time_pct",
            hover_data=["seller_id", "orders_fulfilled"], title="Delivery Speed vs Rating"
        ))

        st.subheader("Sellers to review first")
        show_df(review)
        csv_download(sellers, "Download seller table", "seller_performance.csv", "csv_seller")

        numbers = {
            "sellers_loaded": int(len(sellers)),
            "on_time_target_pct": float(ON_TIME_TARGET),
            "good_rating_target": float(GOOD_RATING),
            "sellers_flagged_for_review": int(sellers["needs_review"].sum()),
            "flagged_seller_revenue": round(
                float(review["revenue"].sum()),
                2,
            ),
            "top_seller_revenue": (
                round(float(sellers.iloc[0]["revenue"]), 2)
                if not sellers.empty
                else None
            ),
            "top_seller_on_time_pct": (
                round(float(sellers.iloc[0]["on_time_pct"]), 2)
                if not sellers.empty
                else None
            ),
            "top_seller_rating": (
                round(float(sellers.iloc[0]["avg_rating"]), 2)
                if not sellers.empty
                else None
            ),
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
        generate_brief_button(
            "New vs Returning Customers",
            numbers,
            status_from_metrics(
                {"returning_share": returning_share}
            ),
        )
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
