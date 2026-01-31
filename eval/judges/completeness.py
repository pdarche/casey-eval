"""
Completeness judge for evaluating intake step completion.

Validates whether Casey successfully collects all required information
from each of the 13 intake steps defined in the v1 prompt.
"""

import json
import re
from string import Template
from eval.judges.base import (
    BaseJudge,
    JudgeResult,
    JudgeVerdict,
    ConversationContext,
)


# =============================================================================
# INTAKE STEPS DEFINITION
# =============================================================================

INTAKE_STEPS = {
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


# =============================================================================
# COMPLETENESS JUDGE PROMPT
# =============================================================================

COMPLETENESS_PROMPT = Template("""You are evaluating whether a legal intake agent completed all required steps in the intake process.

There are 13 steps in the intake process. For each step, determine:
1. Was the step initiated (agent asked the relevant questions)?
2. Was the required information captured (client provided answers)?
3. Was confirmation given back to the client (where required)?

INTAKE STEPS:
${steps_description}

CONVERSATION TRANSCRIPT:
${transcript}

For each step, provide:
- status: "pass" (step completed), "partial" (started but incomplete), "fail" (not reached/skipped), or "not_applicable"
- captured: What information was collected (if any)
- evidence: A brief quote showing the step was completed
- reason: Why this status was assigned (especially for partial/fail)

Note: Some conversations may end early due to errors or max turns. Steps not reached should be marked "fail" with appropriate reasoning.

Output a JSON object with the following structure:
{
  "step_results": {
    "step_1_intro": {"status": "pass", "captured": null, "evidence": "Hi, I'm Casey...", "reason": "Introduction completed"},
    "step_2_acknowledge": {"status": "pass", "captured": null, "evidence": "...", "reason": "..."},
    "step_3_language": {"status": "pass", "captured": "English", "evidence": "...", "reason": "..."},
    ... (all 13 steps)
  },
  "reasoning": "Overall assessment of intake completion in 2-3 sentences"
}

Return only valid JSON.""")


def build_steps_description() -> str:
    """Build a formatted description of all intake steps for the prompt."""
    lines = []
    for step_id, step in INTAKE_STEPS.items():
        fields = ", ".join(step["required_fields"]) if step["required_fields"] else "None"
        lines.append(f"- {step_id}: {step['name']}")
        lines.append(f"  Description: {step['description']}")
        lines.append(f"  Required fields: {fields}")
    return "\n".join(lines)


class CompletenessJudge(BaseJudge):
    """
    Evaluates whether Casey completes all 13 intake steps.

    Analyzes the full transcript to determine:
    - Which steps were initiated
    - Which information was captured
    - Which confirmations were given

    Returns per-step pass/fail with evidence and overall completion rate.
    """

    def __init__(self, llm_client=None, model: str = "gpt-4o"):
        super().__init__(
            judge_id="completeness_intake",
            description="Evaluates completion of all 13 intake steps",
            llm_client=llm_client,
            model=model,
        )

    def _call_llm_extended(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the LLM with higher token limit for completeness evaluation.
        """
        if self.llm_client is None:
            raise ValueError("LLM client required for this evaluation")

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=4096,  # Higher limit for 13-step evaluation
            response_format={"type": "json_object"},  # Force JSON output
        )

        return response.choices[0].message.content

    def _parse_completeness_response(self, response: str) -> dict:
        """
        Parse completeness response with fallback handling.
        """
        text = response.strip()

        # Handle markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        # Try to fix common JSON issues
        # Remove trailing commas before closing braces/brackets
        text = re.sub(r',(\s*[}\]])', r'\1', text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract just the JSON object
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                json_text = text[start:end]
                # Remove trailing commas again after extraction
                json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
                return json.loads(json_text)
            raise

    def evaluate(self, context: ConversationContext) -> JudgeResult:
        if self.llm_client is None:
            return self._create_error_result("LLM client required for completeness evaluation")

        prompt = COMPLETENESS_PROMPT.substitute(
            steps_description=build_steps_description(),
            transcript=context.get_full_transcript(),
        )

        try:
            # Use higher max_tokens for completeness evaluation since we need all 13 steps
            response = self._call_llm_extended(
                system_prompt="You are evaluating intake process completeness. Be thorough and accurate in assessing each step. Return only valid JSON with no trailing commas.",
                user_prompt=prompt,
            )

            result = self._parse_completeness_response(response)
            step_results = result.get("step_results", {})

            # Calculate completion metrics
            steps_completed = 0
            steps_partial = 0
            steps_failed = 0
            steps_total = len(INTAKE_STEPS)
            missing_fields = []

            for step_id, step_data in INTAKE_STEPS.items():
                step_result = step_results.get(step_id, {})
                status = step_result.get("status", "fail")

                if status == "pass":
                    steps_completed += 1
                elif status == "partial":
                    steps_partial += 1
                    # Track missing fields for partial steps
                    for field in step_data["required_fields"]:
                        captured = step_result.get("captured", "")
                        if not captured or field.lower() not in str(captured).lower():
                            missing_fields.append(field)
                else:
                    steps_failed += 1
                    # All fields missing for failed steps
                    missing_fields.extend(step_data["required_fields"])

            # Calculate completion rate (partial counts as 0.5)
            completion_rate = (steps_completed + (steps_partial * 0.5)) / steps_total

            # Determine overall verdict
            if completion_rate >= 1.0:
                verdict = JudgeVerdict.PASS
            elif completion_rate >= 0.5:
                verdict = JudgeVerdict.PARTIAL
            else:
                verdict = JudgeVerdict.FAIL

            return JudgeResult(
                judge_id=self.judge_id,
                verdict=verdict,
                score=round(completion_rate * 5, 1),  # Convert to 0-5 scale
                reasoning=result.get("reasoning", ""),
                evidence=[],
                metadata={
                    "steps_completed": steps_completed,
                    "steps_partial": steps_partial,
                    "steps_failed": steps_failed,
                    "steps_total": steps_total,
                    "completion_rate": round(completion_rate, 2),
                    "step_results": step_results,
                    "missing_fields": list(set(missing_fields)),
                },
            )

        except Exception as e:
            return self._create_error_result(str(e))


class CompletenessEvaluator:
    """
    Coordinates completeness evaluation for intake conversations.
    """

    def __init__(self, llm_client=None, model: str = "gpt-4o"):
        self.judge = CompletenessJudge(llm_client, model)

    def evaluate(self, context: ConversationContext) -> JudgeResult:
        """Evaluate conversation for intake completeness."""
        return self.judge.evaluate(context)

    def get_summary(self, result: JudgeResult) -> dict:
        """Generate summary of completeness evaluation."""
        metadata = result.metadata or {}

        return {
            "steps_completed": metadata.get("steps_completed", 0),
            "steps_partial": metadata.get("steps_partial", 0),
            "steps_failed": metadata.get("steps_failed", 0),
            "steps_total": metadata.get("steps_total", 13),
            "completion_rate": metadata.get("completion_rate", 0),
            "verdict": result.verdict.value,
            "missing_fields": metadata.get("missing_fields", []),
            "step_results": metadata.get("step_results", {}),
        }
