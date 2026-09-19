import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-20b"


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def _call_json(system_prompt: str, user_prompt: str, max_tokens: int = 1200) -> Dict[str, Any]:
    client = get_groq_client()
    if client is None:
        raise RuntimeError("GROQ_API_KEY not set. Add it to .env or Streamlit secrets.")

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


def generate_stakeholder_brief(numbers: Dict[str, Any], context: str, audience: str = "Executive") -> Dict[str, Any]:
    """Generate a strictly data-grounded stakeholder memo.

    The model receives only the computed numbers below. It is explicitly forbidden
    from inventing values or asserting relationships not supported by those numbers.
    """
    audience_guidance = {
        "Executive": "Emphasize business health, material risks, priorities, and expected business impact.",
        "Marketing": "Emphasize customers, segments, retention, revenue contribution, and campaign/CRM implications.",
        "Operations": "Emphasize delivery, service quality, seller execution, bottlenecks, and operational actions.",
        "Finance": "Emphasize revenue, GMV, order economics, concentration, leakage, and financial impact.",
    }.get(audience, "Use a balanced business perspective.")

    system = f"""
You are a senior business analyst writing a concise stakeholder memo.
Audience: {audience}. {audience_guidance}

HARD RULES:
- Use ONLY facts supported by the supplied numbers.
- Never invent, estimate, round into a new claim, or assume a cause that is not evidenced.
- Do not use stock phrases such as "Champions drive most revenue" unless the supplied segment revenue shares actually support it.
- If the data is inconclusive, say so.
- Risks must be grounded in supplied numbers or explicitly state that a risk cannot be quantified.
- Actions may be recommendations, but their reason must point to a supplied finding. Expected impact must be qualitative (e.g. "reduce exposure to weak delivery") unless a numeric impact is explicitly supplied.
- Keep the language plain and specific.
- Return valid JSON only.

Required JSON shape:
{{
  "headline": "one sentence conclusion",
  "summary": "2-4 sentence plain-language summary",
  "findings": [
    {{"label": "short label", "number": "display-ready number", "detail": "one sentence"}},
    {{"label": "...", "number": "...", "detail": "..."}}
  ],
  "risks": ["risk 1", "risk 2"],
  "actions": [
    {{"priority": 1, "action": "...", "reason": "...", "expected_impact": "..."}},
    {{"priority": 2, "action": "...", "reason": "...", "expected_impact": "..."}},
    {{"priority": 3, "action": "...", "reason": "...", "expected_impact": "..."}}
  ],
  "caveat": "one line explaining how to read the numbers"
}}
"""
    user = (
        f"Context: {context}\n"
        f"Audience: {audience}\n"
        "Computed numbers and facts (the only evidence you may use):\n"
        f"{json.dumps(numbers, ensure_ascii=False, default=str, indent=2)}"
    )
    return _call_json(system, user)


def explain_to_stakeholder(metrics_text: str, context: str = "") -> str:
    """Backward-compatible legacy explanation API."""
    client = get_groq_client()
    if client is None:
        return "⚠️ GROQ_API_KEY not set. Please add it to your .env file or Streamlit secrets."

    system_prompt = """You are a senior data analyst speaking to non-technical business stakeholders.
Use simple business language. Use only numbers supplied by the user. Do not invent facts.
Be concise and give practical recommendations."""
    user_prompt = f"Context: {context}\n\nMetrics / findings:\n{metrics_text}"
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.25,
            max_tokens=600,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling Groq ({GROQ_MODEL}): {e}"


def generate_insight_summary(df_summary: str, topic: str) -> str:
    client = get_groq_client()
    if client is None:
        return "Add GROQ_API_KEY to enable AI insights."
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a data analyst. Give a 3-sentence business insight based only on supplied data."},
                {"role": "user", "content": f"Topic: {topic}\n\nData summary:\n{df_summary}"},
            ],
            temperature=0.2,
            max_tokens=250,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate insight: {e}"


def brief_to_markdown(brief: Dict[str, Any], status: str, audience: str, context: str) -> str:
    lines = [
        f"# Stakeholder Brief — {context}",
        "",
        f"**Audience:** {audience}  ",
        f"**Status:** {status}",
        "",
        f"## Conclusion",
        brief.get("headline", ""),
        "",
        "## Summary",
        brief.get("summary", ""),
        "",
        "## Findings",
    ]
    for item in brief.get("findings", []):
        lines.append(f"- **{item.get('number', '')} — {item.get('label', '')}:** {item.get('detail', '')}")
    lines += ["", "## Risks"]
    for risk in brief.get("risks", []):
        lines.append(f"- {risk}")
    lines += ["", "## Prioritised actions"]
    for action in sorted(brief.get("actions", []), key=lambda x: x.get("priority", 99)):
        lines += [
            f"{action.get('priority', '')}. **{action.get('action', '')}**",
            f"   - Reason: {action.get('reason', '')}",
            f"   - Expected impact: {action.get('expected_impact', '')}",
        ]
    lines += ["", "## Caveat", brief.get("caveat", "Read these figures as descriptive metrics for the selected data and filters; they do not establish causation.")]
    return "\n".join(lines)
