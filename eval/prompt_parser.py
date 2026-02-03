"""
Prompt parser for extracting intake steps from Casey prompt content.

Uses an LLM to analyze prompt content and extract structured intake step definitions
that can be used by the completeness judge.
"""

import json
import os
from typing import Optional


def extract_intake_steps(prompt_content: str, llm_client=None, model: str = "gpt-4.1") -> dict:
    """
    Extract intake steps from prompt content using an LLM.

    Args:
        prompt_content: The raw prompt text content
        llm_client: OpenAI client instance (created if not provided)
        model: Model to use for extraction

    Returns:
        Dict with:
        - steps: List of step definitions
        - steps_count: Number of steps found
        - extraction_model: Model used for extraction
    """
    if llm_client is None:
        from openai import OpenAI
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY required for prompt parsing")
        llm_client = OpenAI(api_key=openai_key)

    extraction_prompt = """Analyze this intake agent prompt and extract all the intake steps.

For each step, provide:
- step_id: A snake_case identifier (e.g., "step_1_intro", "step_2_acknowledge")
- name: The step name/title
- description: What happens in this step
- required_fields: List of data fields that must be collected in this step (empty list if none)
- detection_hints: Key phrases that indicate this step is being performed

PROMPT CONTENT:
```
{content}
```

Return a JSON object with this structure:
{{
  "steps": [
    {{
      "step_id": "step_1_intro",
      "name": "Introduction + Warm-Up",
      "description": "Casey introduces itself and asks what's going on for the client",
      "required_fields": [],
      "detection_hints": ["Hi, I'm Casey", "What's going on for you today"]
    }},
    ...
  ]
}}

Extract ALL steps mentioned in the prompt. Be thorough - these steps will be used to evaluate conversation completeness.
Return only valid JSON."""

    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at analyzing intake process prompts and extracting structured step definitions. Return only valid JSON."
            },
            {
                "role": "user",
                "content": extraction_prompt.format(content=prompt_content[:15000])  # Limit content size
            }
        ],
        temperature=0.1,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )

    result_text = response.choices[0].message.content
    result = json.loads(result_text)

    steps = result.get("steps", [])

    # Convert to dict format keyed by step_id for easier lookup
    steps_dict = {}
    for step in steps:
        step_id = step.get("step_id", f"step_{len(steps_dict) + 1}")
        steps_dict[step_id] = {
            "name": step.get("name", ""),
            "description": step.get("description", ""),
            "required_fields": step.get("required_fields", []),
            "detection_hints": step.get("detection_hints", []),
        }

    return {
        "intake_steps": steps_dict,
        "steps_count": len(steps_dict),
        "extraction_model": model,
    }


def get_intake_steps_for_prompt(prompt_version_id: int) -> Optional[dict]:
    """
    Get intake steps for a specific prompt version from the database.

    Args:
        prompt_version_id: The prompt version ID

    Returns:
        Dict of intake steps, or None if not found
    """
    from eval.database import get_prompt_version_by_id

    prompt = get_prompt_version_by_id(prompt_version_id)
    if not prompt:
        return None

    metadata = prompt.metadata or {}
    return metadata.get("intake_steps")


def get_default_intake_steps() -> dict:
    """
    Return the default/fallback intake steps if no prompt-specific steps are available.

    This is the original hardcoded INTAKE_STEPS from completeness.py, kept as fallback.
    """
    return {
        "step_1_intro": {
            "name": "Introduction + Warm-Up",
            "description": "Casey introduces itself and asks 'what's going on for you today?'",
            "required_fields": [],
            "detection_hints": ["Hi, I'm Casey", "What's going on for you today"],
        },
        "step_2_acknowledge": {
            "name": "Acknowledge + Transition",
            "description": "Acknowledges client's response empathetically and transitions to intake",
            "required_fields": [],
            "detection_hints": ["Let's go step by step", "information we need"],
        },
        "step_3_language": {
            "name": "Language Selection",
            "description": "Captures language preference (1 of 11 options)",
            "required_fields": ["language"],
            "detection_hints": ["preferred language", "English", "Spanish", "Mandarin"],
        },
        "step_4_name": {
            "name": "Name Collection",
            "description": "Captures first_name and last_name",
            "required_fields": ["first_name", "last_name"],
            "detection_hints": ["first and last name", "What is your name"],
        },
        "step_5_address": {
            "name": "Address & Eligibility",
            "description": "Captures is_unhoused and full address, confirms back to client",
            "required_fields": ["is_unhoused", "address"],
            "detection_hints": ["unhoused", "address", "street address", "ZIP code"],
        },
        "step_6_disclaimers": {
            "name": "Disclaimers",
            "description": "Presents disclaimer text and captures acknowledgment",
            "required_fields": ["disclaimers_acknowledged"],
            "detection_hints": ["disclaimer", "Open Door Legal", "attorney-client relationship"],
        },
        "step_7_consent": {
            "name": "Terms & Consent",
            "description": "Presents consent terms and captures initials, date, and yes",
            "required_fields": ["initials", "consent_date", "consent_yes"],
            "detection_hints": ["consent", "initials", "CONFIDENTIAL", "18 years of age"],
        },
        "step_8_contact": {
            "name": "Contact Information",
            "description": "Captures date_of_birth, email, and phone",
            "required_fields": ["date_of_birth", "email", "phone"],
            "detection_hints": ["date of birth", "email", "phone number", "contact information"],
        },
        "step_9_demographics": {
            "name": "Communication Demographics",
            "description": "Captures pronouns, primary_language, other_languages, english_fluency",
            "required_fields": ["pronouns", "primary_language", "other_languages", "english_fluency"],
            "detection_hints": ["pronouns", "primary language", "other languages", "English fluency"],
        },
        "step_10_income": {
            "name": "Income & Household",
            "description": "Captures employment, income proof, household details",
            "required_fields": [
                "employment_status",
                "proof_of_income_types",
                "family_type",
                "household_size",
                "num_minors",
                "income_earners",
                "monthly_income",
            ],
            "detection_hints": ["employment", "income", "household", "minors", "family"],
        },
        "step_11_legal": {
            "name": "Legal Issue",
            "description": "Captures legal issue category, summary, parties, and related details",
            "required_fields": [
                "legal_issue_category",
                "legal_issue_summary",
                "other_parties",
                "ipv_related",
                "legal_papers_description",
                "desired_outcome",
                "homelessness_risk",
                "court_deadlines",
            ],
            "detection_hints": ["legal problem", "legal issue", "court deadline", "outcome"],
        },
        "step_12_reporting": {
            "name": "Reporting Demographics",
            "description": "Captures race_ethnicity, gender_identity, sexual_orientation",
            "required_fields": ["race_ethnicity", "gender_identity", "sexual_orientation"],
            "detection_hints": ["race", "ethnicity", "gender identity", "sexual orientation"],
        },
        "step_13_review": {
            "name": "Final Review & Approval",
            "description": "Shows summary and captures submit confirmation",
            "required_fields": ["submit_confirmation"],
            "detection_hints": ["summary", "submit", "review", "INTAKE_COMPLETE"],
        },
    }
