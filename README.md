# Olist SQL Business Analytics Dashboard

**Production-style SQL analytics platform** built the way real Data Scientists work in companies.

## What this project contains

### SQL Analytical Layer (the most important part)
- `vw_order_fact` – clean order-item grain fact table
- `vw_customer_metrics` – lifetime value, AOV, recency
- `vw_rfm_segments` – full RFM scoring + business segments
- `vw_cohort_retention` – classic cohort retention matrix
- `vw_monthly_kpis` – GMV, orders, AOV, on-time %, ratings
- `vw_product_performance` – category revenue & ratings
- `vw_seller_performance` – seller ranking + delivery quality
- `vw_state_performance` – geographic analysis
- `vw_new_vs_returning` – new vs returning revenue split

These are the **exact types of views** Data Scientists and Analytics Engineers create in real companies.

### Streamlit Dashboard
- Executive Overview with KPIs and trends
- Interactive RFM segmentation
- Cohort retention heatmap + curves
- Product, Seller, Geographic analysis
- New vs Returning customer analysis
- **Groq-powered “Explain to Stakeholder”** feature

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd olist-sql-analytics
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# or
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment variables
Create a `.env` file in the root:

```env
DATABASE_URL=postgresql+psycopg://postgres.xxxxx:YOUR_PASSWORD@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres
GROQ_API_KEY=gsk_your_groq_key_here
```

> **Never commit the `.env` file.** It is already in `.gitignore`.

### 5. Create the analytical views in Supabase
1. Open Supabase → SQL Editor
2. Copy the entire content of `sql/01_create_analytics_views.sql`
3. Run it

**Important:** The SQL assumes these table names exist:
- `orders`
- `order_items`
- `customers`
- `products`
- `sellers`
- `order_reviews`
- `product_category_name_translation`

If your tables have different names (e.g. `olist_orders_dataset`), either rename them or edit the view definitions.

### 6. Run locally
```bash
streamlit run app.py
```

---

## Deploy to Streamlit Community Cloud

1. Push the code to GitHub (without `.env`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. In **Secrets** add:

```toml
DATABASE_URL = "postgresql+psycopg://postgres.xxxxx:YOUR_PASSWORD@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres"
GROQ_API_KEY = "gsk_xxxxxxxx"
```

5. Deploy

---

## Project Structure
```
olist-sql-analytics/
├── app.py                          # Main Streamlit application
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── sql/
│   └── 01_create_analytics_views.sql
└── utils/
    ├── db.py                       # Database connection
    └── llm.py                      # Groq stakeholder explanations
```

---

## Skills demonstrated
- Advanced SQL (window functions, CTEs, NTILE, cohort logic, delivery SLAs)
- Analytical view design (star-schema style thinking)
- Business metrics (RFM, retention, GMV, AOV, on-time %)
- Interactive dashboarding with Plotly
- LLM integration for stakeholder communication
- Clean project structure + environment management

---

## Notes for recruiters / hiring managers
This project intentionally focuses on the **SQL + business analytics** skills that most entry-level Data Scientist roles test heavily, while still showing modern tooling (Streamlit + Groq).
