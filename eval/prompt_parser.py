"""
Prompt parser for extracting intake questions from Casey prompt content.

Uses an LLM to analyze prompt content and extract structured per-question definitions
that can be used by the completeness judge.
"""

import json
import os
from typing import Optional


def extract_intake_steps(prompt_content: str, llm_client=None, model: str = "gpt-5-mini") -> dict:
    """
    Extract individual intake questions from prompt content using an LLM.

    Args:
        prompt_content: The raw prompt text content
        llm_client: OpenAI client instance (created if not provided)
        model: Model to use for extraction

    Returns:
        Dict with:
        - intake_steps: Dict of question definitions keyed by question_id
        - steps_count: Number of questions found
        - extraction_model: Model used for extraction
    """
    if llm_client is None:
        from openai import OpenAI
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY required for prompt parsing")
        llm_client = OpenAI(api_key=openai_key)

    extraction_prompt = """Analyze this intake agent prompt and extract every individual QUESTION that the agent asks the client.

The prompt has step_1 through step_13. Each step may contain multiple lettered sub-questions (A, B, C...) or distinct questions. Extract EVERY discrete question, not just the step headers.

For each question, provide:
- question_id: Format "s{{N}}_q{{M}}_{{short_name}}" where N is the step number, M is the question number within that step, and short_name is a snake_case identifier (e.g., "s10_q1_employment_status")
- parent_step: The step number (integer)
- parent_step_name: The step title (e.g., "Income & Household")
- name: Short display name for this question (e.g., "Employment Status")
- question_text: The actual question text Casey asks
- required_field: The single data field this question captures (snake_case, null if no specific field like intro/transition steps)
- detection_hints: Key phrases that indicate this question is being asked or answered

PROMPT CONTENT:
```
{content}
```

Return a JSON object with this structure:
{{
  "questions": [
    {{
      "question_id": "s1_q1_warm_up",
      "parent_step": 1,
      "parent_step_name": "Introduction + Warm-Up",
      "name": "Warm-Up",
      "question_text": "What's going on for you today?",
      "required_field": null,
      "detection_hints": ["Hi, I'm Casey", "What's going on for you today"]
    }},
    ...
  ]
}}

Extract ALL individual questions from ALL steps. Be thorough - steps like Income & Household (step 10) and Legal Issue (step 11) have 8+ sub-questions each.
Return only valid JSON."""

    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at analyzing intake process prompts and extracting structured question definitions. Return only valid JSON."
            },
            {
                "role": "user",
                "content": extraction_prompt.format(content=prompt_content[:15000])
            }
        ],
        max_completion_tokens=8192,
        response_format={"type": "json_object"},
    )

    result_text = response.choices[0].message.content
    result = json.loads(result_text)

    questions = result.get("questions", [])

    # Convert to dict format keyed by question_id for easier lookup
    questions_dict = {}
    for q in questions:
        question_id = q.get("question_id", f"s0_q{len(questions_dict) + 1}_unknown")
        questions_dict[question_id] = {
            "parent_step": q.get("parent_step"),
            "parent_step_name": q.get("parent_step_name", ""),
            "name": q.get("name", ""),
            "question_text": q.get("question_text", ""),
            "required_field": q.get("required_field"),
            "detection_hints": q.get("detection_hints", []),
        }

    return {
        "intake_steps": questions_dict,
        "steps_count": len(questions_dict),
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
    Return the default/fallback intake questions if no prompt-specific steps are available.

    Expanded to per-question granularity (~35-40 questions) derived from casey_prompt_v1.yaml.
    """
    return {
        # Step 1 — Introduction + Warm-Up
        "s1_q1_warm_up": {
            "parent_step": 1,
            "parent_step_name": "Introduction + Warm-Up",
            "name": "Warm-Up",
            "question_text": "What's going on for you today?",
            "required_field": None,
            "detection_hints": ["Hi, I'm Casey", "What's going on for you today"],
        },
        # Step 2 — Acknowledge + Transition (no distinct question, just empathetic transition)
        "s2_q1_acknowledge": {
            "parent_step": 2,
            "parent_step_name": "Acknowledge + Transition",
            "name": "Acknowledge & Transition",
            "question_text": "Acknowledge client's response and transition to intake",
            "required_field": None,
            "detection_hints": ["Let's go step by step", "information we need"],
        },
        # Step 3 — Language Selection
        "s3_q1_language": {
            "parent_step": 3,
            "parent_step_name": "Language Selection",
            "name": "Language",
            "question_text": "Please type one of the following: English, Spanish, Mandarin...",
            "required_field": "language",
            "detection_hints": ["preferred language", "English", "Spanish", "Mandarin"],
        },
        # Step 4 — Name Collection
        "s4_q1_name": {
            "parent_step": 4,
            "parent_step_name": "Name Collection",
            "name": "First & Last Name",
            "question_text": "What is your first and last name? Please type both.",
            "required_field": "first_name",
            "detection_hints": ["first and last name", "What is your name"],
        },
        # Step 5 — Address & Eligibility
        "s5_q1_unhoused": {
            "parent_step": 5,
            "parent_step_name": "Address & Eligibility",
            "name": "Unhoused Status",
            "question_text": "Are you unhoused? Type Yes or No.",
            "required_field": "is_unhoused",
            "detection_hints": ["unhoused", "Are you unhoused"],
        },
        "s5_q2_address": {
            "parent_step": 5,
            "parent_step_name": "Address & Eligibility",
            "name": "Full Address",
            "question_text": "Please type your full street address, city, state, and ZIP code.",
            "required_field": "address",
            "detection_hints": ["street address", "ZIP code", "full address", "city, state"],
        },
        # Step 6 — Disclaimers
        "s6_q1_disclaimers": {
            "parent_step": 6,
            "parent_step_name": "Disclaimers",
            "name": "Disclaimers Acknowledged",
            "question_text": "Do you understand and agree to these disclaimers? Please type Yes or Ask a Question.",
            "required_field": "disclaimers_acknowledged",
            "detection_hints": ["disclaimer", "Open Door Legal", "attorney-client relationship"],
        },
        # Step 7 — Terms & Consent
        "s7_q1_consent": {
            "parent_step": 7,
            "parent_step_name": "Terms & Consent",
            "name": "Consent (Initials, Date, Yes)",
            "question_text": "Please type: 1. Your initials 2. Today's date in MM-DD-YYYY 3. Type Yes to agree",
            "required_field": "consent_yes",
            "detection_hints": ["consent", "initials", "CONFIDENTIAL", "18 years of age"],
        },
        # Step 8 — Contact Information
        "s8_q1_dob": {
            "parent_step": 8,
            "parent_step_name": "Contact Information",
            "name": "Date of Birth",
            "question_text": "What is your date of birth? Please type in MM-DD-YYYY.",
            "required_field": "date_of_birth",
            "detection_hints": ["date of birth", "MM-DD-YYYY", "born"],
        },
        "s8_q2_email": {
            "parent_step": 8,
            "parent_step_name": "Contact Information",
            "name": "Email",
            "question_text": "What is the best email for updates? If none, type No email.",
            "required_field": "email",
            "detection_hints": ["email", "best email"],
        },
        "s8_q3_phone": {
            "parent_step": 8,
            "parent_step_name": "Contact Information",
            "name": "Phone Number",
            "question_text": "What is the best phone number to reach you? If none, type No phone.",
            "required_field": "phone",
            "detection_hints": ["phone number", "best phone"],
        },
        # Step 9 — Communication Demographics
        "s9_q1_pronouns": {
            "parent_step": 9,
            "parent_step_name": "Communication Demographics",
            "name": "Pronouns",
            "question_text": "What gender pronouns do you use? He/Him, She/Her, They/Them, Other, Decline",
            "required_field": "pronouns",
            "detection_hints": ["pronouns", "He/Him", "She/Her", "They/Them"],
        },
        "s9_q2_primary_language": {
            "parent_step": 9,
            "parent_step_name": "Communication Demographics",
            "name": "Primary Language",
            "question_text": "What is your primary language?",
            "required_field": "primary_language",
            "detection_hints": ["primary language"],
        },
        "s9_q3_other_languages": {
            "parent_step": 9,
            "parent_step_name": "Communication Demographics",
            "name": "Other Languages",
            "question_text": "Do you speak any other languages? Please type them, or type None.",
            "required_field": "other_languages",
            "detection_hints": ["other languages", "speak any other"],
        },
        "s9_q4_english_fluency": {
            "parent_step": 9,
            "parent_step_name": "Communication Demographics",
            "name": "English Fluency",
            "question_text": "How would you describe your English fluency? Native, Very Fluent, Fluent, Somewhat Fluent, Not Fluent",
            "required_field": "english_fluency",
            "detection_hints": ["English fluency", "Native", "Very Fluent", "Fluent"],
        },
        # Step 10 — Income & Household
        "s10_q1_employment_status": {
            "parent_step": 10,
            "parent_step_name": "Income & Household",
            "name": "Employment Status",
            "question_text": "Which best describes your employment status? Employed (Full/Part-Time), Unemployed, Disabled, Retired",
            "required_field": "employment_status",
            "detection_hints": ["employment status", "employed", "retired", "unemployed"],
        },
        "s10_q2_income_proof": {
            "parent_step": 10,
            "parent_step_name": "Income & Household",
            "name": "Proof of Income Types",
            "question_text": "Please let us know which proof of income you can provide.",
            "required_field": "proof_of_income_types",
            "detection_hints": ["proof of income", "payroll stub", "CalFresh", "SSI"],
        },
        "s10_q3_income_documents": {
            "parent_step": 10,
            "parent_step_name": "Income & Household",
            "name": "Income Documents Upload",
            "question_text": "Please upload proof of income documents or type Will bring later.",
            "required_field": "income_documents",
            "detection_hints": ["upload proof", "income documents", "Will bring later"],
        },
        "s10_q4_family_type": {
            "parent_step": 10,
            "parent_step_name": "Income & Household",
            "name": "Family Household Type",
            "question_text": "What best describes your family household? Single Headed or Dual Headed Family",
            "required_field": "family_type",
            "detection_hints": ["family household", "Single Headed", "Dual Headed"],
        },
        "s10_q5_household_size": {
            "parent_step": 10,
            "parent_step_name": "Income & Household",
            "name": "Household Size",
            "question_text": "How many total people are in your household, including yourself and minors?",
            "required_field": "household_size",
            "detection_hints": ["household", "total people", "how many"],
        },
        "s10_q6_num_minors": {
            "parent_step": 10,
            "parent_step_name": "Income & Household",
            "name": "Number of Minors",
            "question_text": "How many minors are in your household?",
            "required_field": "num_minors",
            "detection_hints": ["minors", "how many minors"],
        },
        "s10_q7_income_earners": {
            "parent_step": 10,
            "parent_step_name": "Income & Household",
            "name": "Income Earners",
            "question_text": "Who earned income last month? Contact, Spouse/Partner, Another Household Member",
            "required_field": "income_earners",
            "detection_hints": ["earned income", "last month", "spouse", "partner"],
        },
        "s10_q8_monthly_income": {
            "parent_step": 10,
            "parent_step_name": "Income & Household",
            "name": "Monthly Income",
            "question_text": "What was your household pre-tax monthly income last month?",
            "required_field": "monthly_income",
            "detection_hints": ["monthly income", "pre-tax", "income last month"],
        },
        # Step 11 — Legal Issue
        "s11_q1_legal_category": {
            "parent_step": 11,
            "parent_step_name": "Legal Issue",
            "name": "Legal Issue Category",
            "question_text": "What kind of legal problem do you have? Housing, Family Law, Employment, Consumer, etc.",
            "required_field": "legal_issue_category",
            "detection_hints": ["legal problem", "kind of legal", "Housing", "Family Law"],
        },
        "s11_q2_legal_summary": {
            "parent_step": 11,
            "parent_step_name": "Legal Issue",
            "name": "Legal Issue Summary",
            "question_text": "Please summarize your legal issue in 2-3 sentences. Include: who, what, where, when, why.",
            "required_field": "legal_issue_summary",
            "detection_hints": ["summarize", "legal issue", "who, what, where"],
        },
        "s11_q3_other_parties": {
            "parent_step": 11,
            "parent_step_name": "Legal Issue",
            "name": "Other Parties",
            "question_text": "Please list the names of other people involved (people, businesses, agencies).",
            "required_field": "other_parties",
            "detection_hints": ["other people involved", "other parties", "businesses", "agencies"],
        },
        "s11_q4_ipv": {
            "parent_step": 11,
            "parent_step_name": "Legal Issue",
            "name": "Intimate Partner Violence",
            "question_text": "Is your legal issue related to past intimate partner violence? Yes, No, or Prefer not to say.",
            "required_field": "ipv_related",
            "detection_hints": ["intimate partner violence", "IPV", "domestic violence"],
        },
        "s11_q5_legal_papers": {
            "parent_step": 11,
            "parent_step_name": "Legal Issue",
            "name": "Legal Papers",
            "question_text": "Describe any legal papers you have received and the court address if listed.",
            "required_field": "legal_papers_description",
            "detection_hints": ["legal papers", "court address", "papers received"],
        },
        "s11_q6_desired_outcome": {
            "parent_step": 11,
            "parent_step_name": "Legal Issue",
            "name": "Desired Outcome",
            "question_text": "What outcome are you looking for? What would you like done to resolve this problem?",
            "required_field": "desired_outcome",
            "detection_hints": ["outcome", "resolve", "looking for"],
        },
        "s11_q7_homelessness_risk": {
            "parent_step": 11,
            "parent_step_name": "Legal Issue",
            "name": "Homelessness Risk",
            "question_text": "If this issue is not resolved, do you think you may become homeless? Yes or No.",
            "required_field": "homelessness_risk",
            "detection_hints": ["homeless", "become homeless"],
        },
        "s11_q8_court_deadlines": {
            "parent_step": 11,
            "parent_step_name": "Legal Issue",
            "name": "Court Deadlines",
            "question_text": "Do you have any court deadlines or dates? Type them in MM-DD-YYYY or None.",
            "required_field": "court_deadlines",
            "detection_hints": ["court deadline", "court date", "hearing date"],
        },
        # Step 12 — Reporting Demographics
        "s12_q1_race_ethnicity": {
            "parent_step": 12,
            "parent_step_name": "Reporting Demographics",
            "name": "Race/Ethnicity",
            "question_text": "Which best describes your ethnicity or race? Please type one or more from the list.",
            "required_field": "race_ethnicity",
            "detection_hints": ["race", "ethnicity", "Asian", "Black", "Latino", "White"],
        },
        "s12_q2_gender_identity": {
            "parent_step": 12,
            "parent_step_name": "Reporting Demographics",
            "name": "Gender Identity",
            "question_text": "What is your gender? Female, Male, Genderqueer, Non-binary, Trans, Decline",
            "required_field": "gender_identity",
            "detection_hints": ["gender", "gender identity", "Female", "Male", "Non-binary"],
        },
        "s12_q3_sexual_orientation": {
            "parent_step": 12,
            "parent_step_name": "Reporting Demographics",
            "name": "Sexual Orientation",
            "question_text": "How would you describe your sexual orientation or identity?",
            "required_field": "sexual_orientation",
            "detection_hints": ["sexual orientation", "Straight", "Gay", "Bisexual"],
        },
        # Step 13 — Final Review & Approval
        "s13_q1_review_submit": {
            "parent_step": 13,
            "parent_step_name": "Final Review & Approval",
            "name": "Review & Submit",
            "question_text": "Here is your summary. If everything looks correct, type Submit.",
            "required_field": "submit_confirmation",
            "detection_hints": ["summary", "submit", "review", "INTAKE_COMPLETE"],
        },
    }
