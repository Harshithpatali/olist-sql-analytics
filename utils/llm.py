import json
import os
from functools import lru_cache
from typing import Any, Dict

from dotenv import load_dotenv

try:
    from groq import Groq
except ImportError:
    Groq = None


load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

GROQ_MODEL = "openai/gpt-oss-20b"

# Keep the final answer small.
BRIEF_MAX_TOKENS = 1200


# ============================================================
# GROQ CLIENT
# ============================================================

@lru_cache(maxsize=1)
def get_groq_client():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    if Groq is None:
        return None

    return Groq(api_key=api_key)


# ============================================================
# JSON SAFE CONVERSION
# ============================================================

def _json_safe(value):

    try:

        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not serialize AI metrics: {exc}"
        ) from exc


# ============================================================
# STAKEHOLDER BRIEF SCHEMA
# ============================================================

STAKEHOLDER_BRIEF_SCHEMA = {

    "type": "object",

    "properties": {

        "status": {
            "type": "string",
            "enum": [
                "Strong",
                "Stable",
                "Watch closely",
                "At risk",
            ],
        },

        "headline": {
            "type": "string",
        },

        "summary": {
            "type": "string",
        },

        "findings": {

            "type": "array",

            "minItems": 3,
            "maxItems": 3,

            "items": {

                "type": "object",

                "properties": {

                    "label": {
                        "type": "string",
                    },

                    "number": {
                        "type": "string",
                    },

                    "detail": {
                        "type": "string",
                    },

                },

                "required": [
                    "label",
                    "number",
                    "detail",
                ],

                "additionalProperties": False,
            },
        },

        "risks": {

            "type": "array",

            "minItems": 1,
            "maxItems": 2,

            "items": {
                "type": "string",
            },
        },

        "actions": {

            "type": "array",

            "minItems": 3,
            "maxItems": 3,

            "items": {

                "type": "object",

                "properties": {

                    "priority": {
                        "type": "integer",
                        "enum": [1, 2, 3],
                    },

                    "action": {
                        "type": "string",
                    },

                    "reason": {
                        "type": "string",
                    },

                    "expected_impact": {
                        "type": "string",
                    },

                },

                "required": [
                    "priority",
                    "action",
                    "reason",
                    "expected_impact",
                ],

                "additionalProperties": False,
            },
        },

        "caveat": {
            "type": "string",
        },

    },

    "required": [
        "status",
        "headline",
        "summary",
        "findings",
        "risks",
        "actions",
        "caveat",
    ],

    "additionalProperties": False,
}


# ============================================================
# VALIDATION
# ============================================================

def _validate_stakeholder_brief(brief):

    required = {
        "status",
        "headline",
        "summary",
        "findings",
        "risks",
        "actions",
        "caveat",
    }

    missing = required - set(brief.keys())

    if missing:

        raise RuntimeError(
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

        raise RuntimeError(
            f"Invalid status: {brief['status']}"
        )

    if len(brief["findings"]) != 3:

        raise RuntimeError(
            "AI must return exactly 3 findings."
        )

    if len(brief["actions"]) != 3:

        raise RuntimeError(
            "AI must return exactly 3 actions."
        )

    if not 1 <= len(brief["risks"]) <= 2:

        raise RuntimeError(
            "AI must return 1-2 risks."
        )

    return brief


# ============================================================
# STAKEHOLDER BRIEF
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
            "Add GROQ_API_KEY to .env or Streamlit secrets."
        )

    safe_numbers = _json_safe(numbers)


    # --------------------------------------------------------
    # Audience focus
    # --------------------------------------------------------

    audience_focus = {

        "Executive": (
            "business health, revenue, major risks "
            "and strategic priorities"
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

    }

    focus = audience_focus.get(
        audience,
        audience_focus["Executive"],
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Keep instructions short.
    # Do NOT ask GPT-OSS to reason extensively.
    # --------------------------------------------------------

    prompt = f"""
Create a concise stakeholder business brief.

Audience: {audience}

Focus: {focus}

Context:
{context}

Use ONLY the supplied metrics.

Never invent numbers.
Never invent facts.
Do not claim causation.
Keep every text field short.

The output will be validated against a JSON schema.

Metrics:
{json.dumps(
    safe_numbers,
    ensure_ascii=False,
    separators=(",", ":"),
)}
"""


    # --------------------------------------------------------
    # API CALL
    # --------------------------------------------------------

    try:

        response = client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],

            temperature=0,

            # GPT-OSS reasoning can consume completion budget.
            reasoning_effort="low",

            # We do not need reasoning returned to the app.
            include_reasoning=False,

            max_completion_tokens=BRIEF_MAX_TOKENS,

            response_format={

                "type": "json_schema",

                "json_schema": {

                    "name": "stakeholder_brief",

                    "strict": True,

                    "schema": STAKEHOLDER_BRIEF_SCHEMA,

                },
            },
        )

    except Exception as exc:

        raise RuntimeError(
            f"Groq API request failed: {exc}"
        ) from exc


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    if not response.choices:

        raise RuntimeError(
            "Groq returned no choices."
        )


    message = response.choices[0].message

    raw = message.content


    if not raw:

        raise RuntimeError(
            "Groq returned an empty response."
        )


    # --------------------------------------------------------
    # PARSE JSON
    # --------------------------------------------------------

    try:

        brief = json.loads(raw)

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "Groq returned invalid JSON:\n\n"
            + str(raw)
        ) from exc


    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    return _validate_stakeholder_brief(
        brief
    )


# ============================================================
# LEGACY COMPATIBILITY
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


    prompt = f"""
Explain these business analytics to a stakeholder.

Context:
{context}

Metrics:
{metrics_text}

Use only supplied information.
Do not invent numbers.
Be concise.
"""


    try:

        response = client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],

            temperature=0.2,

            reasoning_effort="low",

            include_reasoning=False,

            max_completion_tokens=500,
        )

        return (
            response.choices[0]
            .message
            .content
            or ""
        )

    except Exception as exc:

        return (
            f"Error calling Groq: {exc}"
        )


# ============================================================
# GENERAL INSIGHT SUMMARY
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


    prompt = f"""
Give one concise business insight.

Topic:
{topic}

Data:
{df_summary}

Use only the supplied data.
Do not invent numbers.
"""


    try:

        response = client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],

            temperature=0.2,

            reasoning_effort="low",

            include_reasoning=False,

            max_completion_tokens=200,
        )

        return (
            response.choices[0]
            .message
            .content
            or ""
        )

    except Exception as exc:

        return (
            f"Could not generate insight: {exc}"
        )


# ============================================================
# MARKDOWN EXPORT
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

        brief.get(
            "headline",
            "",
        ),

        "",

        "## Summary",

        brief.get(
            "summary",
            "",
        ),

        "",

        "## Findings",
    ]


    for item in brief.get(
        "findings",
        [],
    ):

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


    for risk in brief.get(
        "risks",
        [],
    ):

        lines.append(
            f"- {risk}"
        )


    lines.extend(
        [
            "",
            "## Prioritised Actions",
        ]
    )


    actions = sorted(
        brief.get(
            "actions",
            [],
        ),
        key=lambda x: x.get(
            "priority",
            99,
        ),
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