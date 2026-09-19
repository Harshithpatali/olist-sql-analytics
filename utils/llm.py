import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Best currently available Groq model for high-quality explanations (Sep 2026)
GROQ_MODEL = "openai/gpt-oss-20b"

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def explain_to_stakeholder(metrics_text: str, context: str = "") -> str:
    """
    Generate a clear, non-technical explanation for business stakeholders.
    Uses Llama 3.3 70B Versatile on Groq.
    """
    client = get_groq_client()
    if client is None:
        return "⚠️ GROQ_API_KEY not set. Please add it to your .env file or Streamlit secrets."

    system_prompt = """You are a senior data analyst speaking to non-technical business stakeholders 
(marketing managers, operations leads, executives).

Rules:
- Use simple, clear business language. Avoid jargon.
- Be concise (4-8 sentences).
- Always end with 2 concrete, actionable recommendations.
- Focus on what the numbers mean for revenue, customers, and operations.
- Do not invent numbers that are not provided.
- Sound confident and professional.
"""

    user_prompt = f"""
Context: {context}

Key Metrics / Findings:
{metrics_text}

Please explain what this means for the business and give 2 practical recommendations.
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.35,
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling Groq ({GROQ_MODEL}): {str(e)}"


def generate_insight_summary(df_summary: str, topic: str) -> str:
    """Shorter version for automatic insights on charts."""
    client = get_groq_client()
    if client is None:
        return "Add GROQ_API_KEY to enable AI insights."

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a data analyst. Give a 3-sentence business insight based on the data. Be specific and actionable."
                },
                {
                    "role": "user",
                    "content": f"Topic: {topic}\n\nData summary:\n{df_summary}"
                }
            ],
            temperature=0.3,
            max_tokens=250
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate insight: {e}"
