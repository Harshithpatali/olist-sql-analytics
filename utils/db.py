import os
from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    """Create SQLAlchemy engine from DATABASE_URL environment variable."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL not found. "
            "Create a .env file or set the environment variable."
        )
    # Ensure we use the psycopg driver
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(database_url, pool_pre_ping=True)


def run_query(query: str, params=None) -> pd.DataFrame:
    """Execute a SQL query and return a pandas DataFrame."""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
    return df


def test_connection() -> bool:
    """Quick connection test."""
    try:
        df = run_query("SELECT 1 AS test")
        return df.iloc[0]["test"] == 1
    except Exception as e:
        print(f"Connection failed: {e}")
        return False
