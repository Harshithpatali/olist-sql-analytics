import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

GROQ_MODEL = "openai/gpt-oss-20b"

# Keep the brief deliberately compact.
# The previous implementation could hit the completion limit
# while trying to produce the required JSON.
BRIEF_MAX_TOKENS = 1800


# ------------------------------------------------------------
# Groq client
# ------------------------------------------------------------

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ------------------------------------------------------------
# JSON helper
# ------------------------------------------------------------

def _call_json(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = BRIEF_MAX_TOKENS,
) -> Dict[str, Any]:

    client = get_groq_client()

    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY not set. "
            "Add it to .env or Streamlit secrets."
        )

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
        temperature=0.1,
        max_tokens=max_tokens,
        response_format={
            "type": "json_object"
        },
    )

    if not response.choices:
        raise RuntimeError("Groq returned no choices.")

    raw = response.choices[0].message.content

    if not raw:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    try:
        return json.loads(raw)

    except json.JSONDecodeError as e:
        raise RuntimeError(
            "Groq returned invalid JSON. "
            f"Raw response: {raw[:1000]}"
        ) from e


# ------------------------------------------------------------
# Validate stakeholder brief
# ------------------------------------------------------------

def _validate_stakeholder_brief(
    brief: Dict[str, Any]
) -> Dict[str, Any]:

    required_keys = {
        "status",
        "headline",
        "summary",
        "findings",
        "risks",
        "actions",
        "caveat",
    }

    missing = required_keys - set(brief.keys())

    if missing:
        raise ValueError(
            "Stakeholder brief is missing required fields: "
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
            f"Invalid status '{brief['status']}'. "
            f"Expected one of: {', '.join(sorted(valid_statuses))}"
        )

    if not isinstance(brief["findings"], list):
        raise ValueError(
            "Brief findings must be a list."
        )

    if not 3 <= len(brief["findings"]) <= 4:
        raise ValueError(
            "Brief must contain 3 or 4 findings."
        )

    for i, finding in enumerate(brief["findings"]):

        if not isinstance(finding, dict):
            raise ValueError(
                f"Finding {i + 1} must be an object."
            )

        for key in [
            "label",
            "number",
            "detail",
        ]:
            if key not in finding:
                raise ValueError(
                    f"Finding {i + 1} is missing '{key}'."
                )

    if not isinstance(brief["risks"], list):
        raise ValueError(
            "Brief risks must be a list."
        )

    if len(brief["risks"]) > 3:
        brief["risks"] = brief["risks"][:3]

    if not isinstance(brief["actions"], list):
        raise ValueError(
            "Brief actions must be a list."
        )

    if len(brief["actions"]) != 3:
        raise ValueError(
            "Brief must contain exactly 3 actions."
        )

    for i, action in enumerate(brief["actions"]):

        if not isinstance(action, dict):
            raise ValueError(
                f"Action {i + 1} must be an object."
            )

        for key in [
            "priority",
            "action",
            "reason",
            "expected_impact",
        ]:
            if key not in action:
                raise ValueError(
                    f"Action {i + 1} is missing '{key}'."
                )

    return brief


# ------------------------------------------------------------
# Stakeholder brief
# ------------------------------------------------------------

def generate_stakeholder_brief(
    numbers: Dict[str, Any],
    context: str,
    audience: str = "Executive",
) -> Dict[str, Any]:

    """
    Generate a concise, data-grounded stakeholder brief.

    The LLM receives only the computed metrics supplied by the
    application. It does not calculate business metrics itself.
    """

    audience_guidance = {

        "Executive": (
            "Focus on overall business health, material risks, "
            "revenue, customer health and strategic priorities."
        ),

        "Marketing": (
            "Focus on customers, segments, retention, "
            "repeat purchasing, revenue contribution and "
            "customer growth opportunities."
        ),

        "Operations": (
            "Focus on delivery, service quality, sellers, "
            "operational weaknesses and execution priorities."
        ),

        "Finance": (
            "Focus on revenue, order economics, customer value, "
            "revenue concentration and financial risks."
        ),

    }.get(
        audience,
        "Use a balanced business perspective."
    )

    # --------------------------------------------------------
    # Convert everything into JSON-safe values
    # --------------------------------------------------------

    try:
        safe_numbers = json.loads(
            json.dumps(
                numbers,
                ensure_ascii=False,
                default=str,
            )
        )
    except Exception as e:
        raise ValueError(
            f"Could not serialize stakeholder metrics: {e}"
        )

    # --------------------------------------------------------
    # Compact system prompt
    # --------------------------------------------------------

    system_prompt = f"""
You are a senior business analyst.

Audience: {audience}

Audience focus:
{audience_guidance}

Use ONLY the supplied metrics.

Never invent numbers.
Never invent facts.
Do not claim causation unless the supplied data proves it.
Do not assume that a segment is strong or weak without evidence.
Recommendations may be qualitative, but their reasons must reference supplied metrics.

Return ONLY valid JSON.

Use exactly this structure:

{{
  "status": "Strong",
  "headline": "one concise sentence",
  "summary": "two concise sentences",
  "findings": [
    {{
      "label": "short label",
      "number": "display-ready number",
      "detail": "one concise sentence"
    }},
    {{
      "label": "short label",
      "number": "display-ready number",
      "detail": "one concise sentence"
    }},
    {{
      "label": "short label",
      "number": "display-ready number",
      "detail": "one concise sentence"
    }}
  ],
  "risks": [
    "short evidence-based risk",
    "short evidence-based risk"
  ],
  "actions": [
    {{
      "priority": 1,
      "action": "short action",
      "reason": "short evidence-based reason",
      "expected_impact": "short qualitative impact"
    }},
    {{
      "priority": 2,
      "action": "short action",
      "reason": "short evidence-based reason",
      "expected_impact": "short qualitative impact"
    }},
    {{
      "priority": 3,
      "action": "short action",
      "reason": "short evidence-based reason",
      "expected_impact": "short qualitative impact"
    }}
  ],
  "caveat": "one concise sentence"
}}

Rules:
- status must be exactly one of:
  Strong
  Stable
  Watch closely
  At risk
- exactly 3 findings
- maximum 2 risks
- exactly 3 actions
- keep all text concise
- no markdown
- no bullet points
- no commentary outside JSON
"""

    # --------------------------------------------------------
    # Compact user prompt
    # --------------------------------------------------------

    user_prompt = f"""
Business context:
{context}

Computed metrics:
{json.dumps(
    safe_numbers,
    ensure_ascii=False,
    indent=2,
)}

Create the stakeholder brief now.
Return JSON only.
"""

    # --------------------------------------------------------
    # Call model
    # --------------------------------------------------------

    brief = _call_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=BRIEF_MAX_TOKENS,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    return _validate_stakeholder_brief(brief)


# ------------------------------------------------------------
# Legacy compatibility function
# ------------------------------------------------------------

def explain_to_stakeholder(
    metrics_text: str,
    context: str = "",
) -> str:

    """
    Backward-compatible legacy explanation API.

    Existing code using explain_to_stakeholder()
    continues to work.
    """

    client = get_groq_client()

    if client is None:
        return (
            "⚠️ GROQ_API_KEY not set. "
            "Please add it to your .env file or "
            "Streamlit secrets."
        )

    system_prompt = """
You are a senior data analyst speaking to
non-technical business stakeholders.

Use simple business language.

Use only numbers supplied by the user.

Do not invent facts.

Do not invent numbers.

Be concise.

Give practical recommendations when supported
by the supplied metrics.
"""

    user_prompt = (
        f"Context: {context}\n\n"
        f"Metrics / findings:\n{metrics_text}"
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
            max_tokens=600,
        )

        return response.choices[0].message.content

    except Exception as e:

        return (
            f"Error calling Groq "
            f"({GROQ_MODEL}): {e}"
        )


# ------------------------------------------------------------
# General insight summary
# ------------------------------------------------------------

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
                        "Give a concise business insight "
                        "based only on supplied data."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Topic: {topic}\n\n"
                        f"Data summary:\n{df_summary}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=250,
        )

        return response.choices[0].message.content

    except Exception as e:

        return (
            f"Could not generate insight: {e}"
        )


# ------------------------------------------------------------
# Convert brief to Markdown
# ------------------------------------------------------------

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
            "## Prioritised actions",
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
                    "Read these figures as descriptive "
                    "metrics for the selected data and "
                    "filters; they do not establish causation."
                ),
            ),
        ]
    )

    return "\n".join(lines)