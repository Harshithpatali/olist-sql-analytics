-- ============================================================
-- Olist SQL Analytics Layer
-- Uses your existing fact_order_items + dim tables
-- Run this entire script in Supabase SQL Editor
-- ============================================================

-- 1. Main Fact View (adds customer_unique_id + clean time columns)
CREATE OR REPLACE VIEW vw_order_fact AS
SELECT
    f.order_id,
    f.order_item_id,
    f.customer_id,
    c.customer_unique_id,
    c.customer_city,
    c.customer_state,
    f.product_id,
    f.seller_id,
    f.order_status,
    f.order_purchase_timestamp,
    f.order_approved_at,
    f.order_delivered_carrier_date,
    f.order_delivered_customer_date,
    f.order_estimated_delivery_date,
    f.purchase_date,
    f.purchase_year,
    f.purchase_month,
    f.purchase_ym AS year_month,
    DATE_TRUNC('month', f.order_purchase_timestamp)::date AS purchase_month_date,
    f.price,
    f.freight_value,
    f.item_total AS total_item_value,
    f.delivery_days,
    f.delivery_status,
    f.days_vs_estimate,
    f.avg_review_score AS review_score,
    f.category_en AS product_category_english,
    f.primary_payment_type,
    f.max_installments,
    f.total_payment_value,
    s.seller_city,
    s.seller_state
FROM fact_order_items f
LEFT JOIN dim_customers c 
    ON f.customer_id = c.customer_id
LEFT JOIN dim_sellers s 
    ON f.seller_id = s.seller_id
WHERE f.order_status = 'delivered';
;


-- 2. Customer Lifetime Metrics
CREATE OR REPLACE VIEW vw_customer_metrics AS
WITH customer_orders AS (
    SELECT
        customer_unique_id,
        order_id,
        order_purchase_timestamp,
        SUM(total_item_value) AS order_value
    FROM vw_order_fact
    WHERE customer_unique_id IS NOT NULL
    GROUP BY 1, 2, 3
)
SELECT
    customer_unique_id,
    COUNT(DISTINCT order_id) AS total_orders,
    SUM(order_value) AS total_revenue,
    AVG(order_value) AS avg_order_value,
    MIN(order_purchase_timestamp) AS first_order_date,
    MAX(order_purchase_timestamp) AS last_order_date,
    EXTRACT(DAY FROM (MAX(order_purchase_timestamp) - MIN(order_purchase_timestamp))) AS customer_lifetime_days,
    EXTRACT(DAY FROM (
        (SELECT MAX(order_purchase_timestamp) FROM vw_order_fact) - MAX(order_purchase_timestamp)
    )) AS recency_days
FROM customer_orders
GROUP BY customer_unique_id
;


-- 3. RFM Segmentation
CREATE OR REPLACE VIEW vw_rfm_segments AS
WITH rfm_scores AS (
    SELECT
        customer_unique_id,
        total_orders,
        total_revenue,
        avg_order_value,
        recency_days,
        first_order_date,
        last_order_date,
        NTILE(5) OVER (ORDER BY recency_days ASC)  AS r_score,
        NTILE(5) OVER (ORDER BY total_orders DESC) AS f_score,
        NTILE(5) OVER (ORDER BY total_revenue DESC) AS m_score
    FROM vw_customer_metrics
)
SELECT
    *,
    (r_score + f_score + m_score) AS rfm_total,
    CASE
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
        WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN 'Loyal Customers'
        WHEN r_score >= 4 AND f_score <= 2 THEN 'Promising / New'
        WHEN r_score <= 2 AND f_score >= 3 AND m_score >= 3 THEN 'At Risk'
        WHEN r_score <= 2 AND f_score <= 2 THEN 'Lost / Hibernating'
        WHEN m_score >= 4 THEN 'Big Spenders'
        ELSE 'Need Attention'
    END AS rfm_segment
FROM rfm_scores
;


-- 4. Cohort Retention
CREATE OR REPLACE VIEW vw_cohort_retention AS
WITH first_orders AS (
    SELECT
        customer_unique_id,
        DATE_TRUNC('month', MIN(order_purchase_timestamp))::date AS cohort_month
    FROM vw_order_fact
    WHERE customer_unique_id IS NOT NULL
    GROUP BY 1
),
customer_activity AS (
    SELECT
        f.customer_unique_id,
        f.cohort_month,
        DATE_TRUNC('month', o.order_purchase_timestamp)::date AS activity_month
    FROM first_orders f
    JOIN vw_order_fact o ON f.customer_unique_id = o.customer_unique_id
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(DISTINCT customer_unique_id) AS cohort_size
    FROM first_orders
    GROUP BY 1
),
retention AS (
    SELECT
        cohort_month,
        activity_month,
        COUNT(DISTINCT customer_unique_id) AS active_customers,
        (EXTRACT(YEAR FROM age(activity_month, cohort_month)) * 12 +
         EXTRACT(MONTH FROM age(activity_month, cohort_month)))::int AS month_number
    FROM customer_activity
    GROUP BY 1, 2
)
SELECT
    r.cohort_month,
    r.month_number,
    r.active_customers,
    c.cohort_size,
    ROUND(100.0 * r.active_customers / NULLIF(c.cohort_size, 0), 2) AS retention_rate
FROM retention r
JOIN cohort_sizes c ON r.cohort_month = c.cohort_month
WHERE r.month_number >= 0
ORDER BY 1, 2
;


-- 5. Monthly Business KPIs
CREATE OR REPLACE VIEW vw_monthly_kpis AS
SELECT
    purchase_month_date AS purchase_month,
    year_month,
    COUNT(DISTINCT order_id) AS total_orders,
    COUNT(DISTINCT customer_unique_id) AS unique_customers,
    SUM(total_item_value) AS gmv,
    AVG(total_item_value) AS aov,
    SUM(price) AS product_revenue,
    SUM(freight_value) AS freight_revenue,
    AVG(review_score) AS avg_review_score,
    AVG(delivery_days) AS avg_delivery_days,
    COUNT(*) FILTER (WHERE delivery_status = 'On Time') * 100.0 
        / NULLIF(COUNT(*), 0) AS on_time_pct
FROM vw_order_fact
GROUP BY 1, 2
ORDER BY 1
;


-- 6. Product & Category Performance
CREATE OR REPLACE VIEW vw_product_performance AS
SELECT
    product_category_english AS category,
    COUNT(DISTINCT order_id) AS orders,
    COUNT(*) AS items_sold,
    SUM(price) AS revenue,
    SUM(freight_value) AS total_freight,
    AVG(price) AS avg_price,
    AVG(review_score) AS avg_rating,
    COUNT(DISTINCT customer_unique_id) AS unique_buyers
FROM vw_order_fact
WHERE product_category_english IS NOT NULL
GROUP BY 1
ORDER BY revenue DESC
;


-- 7. Seller Performance
CREATE OR REPLACE VIEW vw_seller_performance AS
SELECT
    seller_id,
    seller_city,
    seller_state,
    COUNT(DISTINCT order_id) AS orders_fulfilled,
    COUNT(*) AS items_sold,
    SUM(price) AS revenue,
    AVG(review_score) AS avg_rating,
    AVG(delivery_days) AS avg_delivery_days,
    COUNT(*) FILTER (WHERE delivery_status = 'On Time') * 100.0 
        / NULLIF(COUNT(*), 0) AS on_time_pct
FROM vw_order_fact
GROUP BY 1, 2, 3
ORDER BY revenue DESC
;


-- 8. Geographic / State Performance
CREATE OR REPLACE VIEW vw_state_performance AS
SELECT
    customer_state,
    COUNT(DISTINCT order_id) AS orders,
    COUNT(DISTINCT customer_unique_id) AS customers,
    SUM(total_item_value) AS gmv,
    AVG(total_item_value) AS aov,
    AVG(delivery_days) AS avg_delivery_days,
    AVG(review_score) AS avg_rating,
    COUNT(*) FILTER (WHERE delivery_status = 'On Time') * 100.0 
        / NULLIF(COUNT(*), 0) AS on_time_pct
FROM vw_order_fact
WHERE customer_state IS NOT NULL
GROUP BY 1
ORDER BY gmv DESC
;


-- 9. New vs Returning Customers
CREATE OR REPLACE VIEW vw_new_vs_returning AS
WITH first_order AS (
    SELECT
        customer_unique_id,
        MIN(purchase_month_date) AS first_month
    FROM vw_order_fact
    WHERE customer_unique_id IS NOT NULL
    GROUP BY 1
),
monthly AS (
    SELECT
        f.purchase_month_date AS purchase_month,
        f.customer_unique_id,
        CASE WHEN f.purchase_month_date = fo.first_month THEN 'New' ELSE 'Returning' END AS customer_type,
        f.total_item_value
    FROM vw_order_fact f
    JOIN first_order fo ON f.customer_unique_id = fo.customer_unique_id
)
SELECT
    purchase_month,
    customer_type,
    COUNT(DISTINCT customer_unique_id) AS customers,
    COUNT(*) AS items,
    SUM(total_item_value) AS revenue
FROM monthly
GROUP BY 1, 2
ORDER BY 1, 2
;


-- 10. Payment Analysis
CREATE OR REPLACE VIEW vw_payment_analysis AS
SELECT
    payment_type,
    COUNT(DISTINCT order_id) AS orders,
    AVG(payment_installments) AS avg_installments,
    SUM(payment_value) AS total_paid,
    AVG(payment_value) AS avg_payment
FROM olist_order_payments
GROUP BY 1
ORDER BY total_paid DESC
;


-- 11. Data Quality Check
CREATE OR REPLACE VIEW vw_data_quality AS
SELECT 'Total delivered order items' AS metric, COUNT(*)::text AS value FROM vw_order_fact
UNION ALL
SELECT 'Unique customers', COUNT(DISTINCT customer_unique_id)::text FROM vw_order_fact
UNION ALL
SELECT 'Date range start', MIN(order_purchase_timestamp)::text FROM vw_order_fact
UNION ALL
SELECT 'Date range end', MAX(order_purchase_timestamp)::text FROM vw_order_fact
UNION ALL
SELECT 'Categories with data', COUNT(DISTINCT product_category_english)::text FROM vw_order_fact
;
