import json
import os
from functools import lru_cache
from typing import Any, Dict

from dotenv import load_dotenv
try:
    from groq import Groq
except ImportError:  # Keep the app importable when the optional SDK is absent.
    Groq = None

load_dotenv()

# ============================================================
# Configuration
# ============================================================

GROQ_MODEL = "openai/gpt-oss-20b"

# The brief is intentionally short. A larger limit is retained
# because reasoning-capable models can consume completion budget.
BRIEF_MAX_TOKENS = 2400


# ============================================================
# Groq client
# ============================================================

@lru_cache(maxsize=1)
def get_groq_client():
    """
    Cache the Groq client for the lifetime of the Streamlit
    process. This avoids rebuilding the client on every rerun.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or Groq is None:
        return None

    return Groq(api_key=api_key)


# ============================================================
# JSON utilities
# ============================================================

def _json_safe(value):
    """Convert pandas/numpy/date-like values to JSON-safe data."""
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )
        )
    except Exception as exc:
        raise ValueError(
            f"Could not serialize AI metrics: {exc}"
        ) from exc


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Parse a JSON object even if a model accidentally wraps it
    in a markdown code fence or adds surrounding whitespace.
    """
    if not text:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.replace(
            "```json",
            "",
            1,
        ).replace(
            "```",
            "",
        ).strip()

    try:
        result = json.loads(cleaned)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # Last-resort extraction of the outermost JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start >= 0 and end > start:
        candidate = cleaned[start:end + 1]

        try:
            result = json.loads(candidate)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

    raise RuntimeError(
        "Groq returned invalid JSON.\n\n"
        f"Raw response:\n{cleaned[:4000]}"
    )


# ============================================================
# Stakeholder brief validation
# ============================================================

def _validate_stakeholder_brief(
    brief: Dict[str, Any],
) -> Dict[str, Any]:

    required = {
        "status",
        "headline",
        "summary",
        "findings",
        "risks",
        "actions",
        "caveat",
    }

    missing = required - set(brief)

    if missing:
        raise ValueError(
            "AI brief is missing: "
            + ", ".join(sorted(missing))
        )

    valid_statuses = {
        "Strong",
        "Stable",
        "Watch closely",
        "At risk",
    }

    if brief["status"] not in valid_statuses:
        raise ValueError(
            "Invalid AI status: "
            f"{brief['status']}"
        )

    findings = brief["findings"]

    if not isinstance(findings, list):
        raise ValueError(
            "AI findings must be a list."
        )

    if not 3 <= len(findings) <= 4:
        raise ValueError(
            f"AI returned {len(findings)} findings; "
            "expected 3 or 4."
        )

    for i, item in enumerate(findings, 1):

        if not isinstance(item, dict):
            raise ValueError(
                f"Finding {i} is not an object."
            )

        for key in (
            "label",
            "number",
            "detail",
        ):
            if key not in item:
                raise ValueError(
                    f"Finding {i} missing '{key}'."
                )

    risks = brief["risks"]

    if not isinstance(risks, list):
        raise ValueError(
            "AI risks must be a list."
        )

    brief["risks"] = [
        str(x) for x in risks[:2]
    ]

    actions = brief["actions"]

    if not isinstance(actions, list):
        raise ValueError(
            "AI actions must be a list."
        )

    if len(actions) != 3:
        raise ValueError(
            f"AI returned {len(actions)} actions; "
            "expected exactly 3."
        )

    for i, item in enumerate(actions, 1):

        if not isinstance(item, dict):
            raise ValueError(
                f"Action {i} is not an object."
            )

        for key in (
            "priority",
            "action",
            "reason",
            "expected_impact",
        ):
            if key not in item:
                raise ValueError(
                    f"Action {i} missing '{key}'."
                )

    return brief


# ============================================================
# Main stakeholder brief
# ============================================================

def generate_stakeholder_brief(
    numbers: Dict[str, Any],
    context: str,
    audience: str = "Executive",
) -> Dict[str, Any]:

    client = get_groq_client()

    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add it to .env or Streamlit secrets."
        )

    safe_numbers = _json_safe(numbers)

    focus = {
        "Executive": (
            "business health, material risks, "
            "revenue and strategic priorities"
        ),
        "Marketing": (
            "customers, segments, retention, "
            "repeat purchasing and growth"
        ),
        "Operations": (
            "delivery, sellers, service quality "
            "and operational reliability"
        ),
        "Finance": (
            "revenue, order economics, customer value "
            "and financial exposure"
        ),
    }.get(
        audience,
        "overall business performance",
    )

    # Keep this prompt deliberately compact.
    system_prompt = """
You are a concise business analytics assistant.

Use ONLY the supplied metrics.
Never invent numbers or facts.
Do not claim causation.
Return ONLY valid JSON.
Keep every text field short.
"""

    user_prompt = f"""
Create a stakeholder brief for the {audience} audience.

Focus: {focus}

Context:
{context}

Return exactly this JSON structure:

{{
  "status": "Stable",
  "headline": "one sentence",
  "summary": "two short sentences",
  "findings": [
    {{
      "label": "short label",
      "number": "number",
      "detail": "short explanation"
    }},
    {{
      "label": "short label",
      "number": "number",
      "detail": "short explanation"
    }},
    {{
      "label": "short label",
      "number": "number",
      "detail": "short explanation"
    }}
  ],
  "risks": [
    "short evidence-based risk"
  ],
  "actions": [
    {{
      "priority": 1,
      "action": "short action",
      "reason": "short reason",
      "expected_impact": "short impact"
    }},
    {{
      "priority": 2,
      "action": "short action",
      "reason": "short reason",
      "expected_impact": "short impact"
    }},
    {{
      "priority": 3,
      "action": "short action",
      "reason": "short reason",
      "expected_impact": "short impact"
    }}
  ],
  "caveat": "one short caveat"
}}

Status must be exactly:
Strong, Stable, Watch closely, or At risk.

Exactly 3 findings.
Exactly 3 actions.
Maximum 2 risks.

Metrics:
{json.dumps(
    safe_numbers,
    ensure_ascii=False,
    separators=(",", ":"),
)}

Return JSON only.
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
            max_tokens=BRIEF_MAX_TOKENS,
            response_format={
                "type": "json_object"
            },
        )

    except Exception as exc:
        raise RuntimeError(
            f"Groq API request failed: {exc}"
        ) from exc

    if not response.choices:
        raise RuntimeError(
            "Groq returned no choices."
        )

    raw = response.choices[0].message.content

    brief = _extract_json_object(raw)

    return _validate_stakeholder_brief(brief)


# ============================================================
# Legacy compatibility
# ============================================================

def explain_to_stakeholder(
    metrics_text: str,
    context: str = "",
) -> str:

    client = get_groq_client()

    if client is None:
        return (
            "GROQ_API_KEY is not configured."
        )

    system_prompt = """
You are a business analyst.

Explain the supplied metrics in simple language.
Use only supplied information.
Do not invent numbers.
Be concise.
"""

    user_prompt = (
        f"Context: {context}\n\n"
        f"Metrics:\n{metrics_text}"
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.2,
            max_tokens=500,
        )

        return (
            response.choices[0]
            .message
            .content
        )

    except Exception as exc:
        return (
            f"Error calling Groq: {exc}"
        )


# ============================================================
# General insight summary
# ============================================================

def generate_insight_summary(
    df_summary: str,
    topic: str,
) -> str:

    client = get_groq_client()

    if client is None:
        return (
            "Add GROQ_API_KEY to enable AI insights."
        )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data analyst. "
                        "Give one concise business insight "
                        "based only on supplied data."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Topic: {topic}\n\n"
                        f"Data:\n{df_summary}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=200,
        )

        return (
            response.choices[0]
            .message
            .content
        )

    except Exception as exc:
        return (
            f"Could not generate insight: {exc}"
        )


# ============================================================
# Markdown export
# ============================================================

def brief_to_markdown(
    brief: Dict[str, Any],
    status: str,
    audience: str,
    context: str,
) -> str:

    lines = [
        f"# Stakeholder Brief — {context}",
        "",
        f"**Audience:** {audience}",
        f"**Status:** {status}",
        "",
        "## Conclusion",
        brief.get("headline", ""),
        "",
        "## Summary",
        brief.get("summary", ""),
        "",
        "## Findings",
    ]

    for item in brief.get("findings", []):

        lines.append(
            "- **"
            f"{item.get('number', '')}"
            " — "
            f"{item.get('label', '')}"
            ":** "
            f"{item.get('detail', '')}"
        )

    lines.extend(
        [
            "",
            "## Risks",
        ]
    )

    for risk in brief.get("risks", []):
        lines.append(f"- {risk}")

    lines.extend(
        [
            "",
            "## Prioritised Actions",
        ]
    )

    actions = sorted(
        brief.get("actions", []),
        key=lambda x: x.get("priority", 99),
    )

    for action in actions:

        lines.extend(
            [
                (
                    f"{action.get('priority', '')}. "
                    f"**{action.get('action', '')}**"
                ),
                (
                    "   - Reason: "
                    f"{action.get('reason', '')}"
                ),
                (
                    "   - Expected impact: "
                    f"{action.get('expected_impact', '')}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Caveat",
            brief.get(
                "caveat",
                (
                    "These figures describe the selected "
                    "data and filters; they do not establish causation."
                ),
            ),
        ]
    )

    return "\n".join(lines)
