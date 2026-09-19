# Quick Setup Notes

## 1. Table Name Check (Very Important)

Your Supabase may have tables named differently.  
Run this in Supabase SQL Editor first:

```sql
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;
```

Common possibilities:
- `orders` / `olist_orders_dataset`
- `order_items` / `olist_order_items_dataset`
- `customers` / `olist_customers_dataset`
- etc.

If the names are different, open `sql/01_create_analytics_views.sql` and replace the table names accordingly.

## 2. Create the views

After confirming table names, run the full content of:
`sql/01_create_analytics_views.sql`

## 3. Local .env file

Create a file named `.env` in the project root with:

```
DATABASE_URL=postgresql+psycopg://postgres.njljkgmxjhvbnevbegoc:Harshi%4012345677@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres
GROQ_API_KEY=your_groq_key_here
```

(Use the exact URL you already have.)

## 4. Get a free Groq API key

1. Go to https://console.groq.com
2. Create an account
3. Generate an API key
4. Paste it into the `.env` file

## 5. Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
