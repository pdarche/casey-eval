"""
Completeness judge for evaluating intake question completion.

Validates whether Casey successfully collects all required information
from the intake questions defined in the prompt being evaluated.
"""

import json
import re
from string import Template
from typing import Optional
from eval.judges.base import (
    BaseJudge,
    JudgeResult,
    JudgeVerdict,
    ConversationContext,
)


def get_intake_steps(prompt_version_id: Optional[int] = None) -> dict:
    """
    Get intake steps/questions for evaluation.

    If a prompt_version_id is provided, attempts to load steps from that prompt's metadata.
    Falls back to default steps if not available.

    Args:
        prompt_version_id: Optional prompt version ID to load steps from

    Returns:
        Dict of intake steps keyed by step_id/question_id
    """
    if prompt_version_id is not None:
        try:
            from eval.prompt_parser import get_intake_steps_for_prompt
            steps = get_intake_steps_for_prompt(prompt_version_id)
            if steps:
                return steps
        except Exception as e:
            print(f"Warning: Failed to load prompt-specific intake steps: {e}")

    # Fall back to default steps
    from eval.prompt_parser import get_default_intake_steps
    return get_default_intake_steps()


def _is_new_format(intake_steps: dict) -> bool:
    """Check if intake steps use the new per-question format (has parent_step key)."""
    for step_data in intake_steps.values():
        return "parent_step" in step_data
    return False


# =============================================================================
# COMPLETENESS JUDGE PROMPT
# =============================================================================

COMPLETENESS_PROMPT = Template("""You are evaluating whether a legal intake agent completed all required intake questions.

For each question listed below, determine:
1. Did Casey ask this question (or convey its intent)?
2. Did the client provide a response?
3. Was the relevant information captured?

INTAKE QUESTIONS:
${steps_description}

CONVERSATION TRANSCRIPT:
${transcript}

For each question, provide:
- status: "pass" (question asked and answered), "partial" (asked but incomplete/unclear answer), "fail" (not asked or skipped), or "not_applicable"
- captured: What information was collected (if any)
- evidence: A brief quote showing the question was asked or answered
- reason: Why this status was assigned (especially for partial/fail)

Note: Some conversations may end early due to errors or max turns. Questions not reached should be marked "fail" with appropriate reasoning.

Output a JSON object with the following structure:
{
  "step_results": {
    "<question_id>": {"status": "pass", "captured": null, "evidence": "...", "reason": "..."},
    ... (one entry for each question listed above)
  },
  "reasoning": "Overall assessment of intake completion in 2-3 sentences"
}

Return only valid JSON.""")


def build_steps_description(intake_steps: dict) -> str:
    """Build a formatted description of all intake questions for the prompt.

    Auto-detects format: if entries have 'parent_step' key, groups by parent step.
    Otherwise, falls back to flat list (old format).
    """
    if _is_new_format(intake_steps):
        return _build_grouped_description(intake_steps)
    return _build_flat_description(intake_steps)


def _build_grouped_description(intake_steps: dict) -> str:
    """Build description grouped by parent step (new format)."""
    # Group questions by parent_step
    groups = {}
    for q_id, q_data in intake_steps.items():
        parent = q_data.get("parent_step", 0)
        if parent not in groups:
            groups[parent] = {
                "name": q_data.get("parent_step_name", f"Step {parent}"),
                "questions": [],
            }
        groups[parent]["questions"].append((q_id, q_data))

    lines = []
    for step_num in sorted(groups.keys()):
        group = groups[step_num]
        lines.append(f"Step {step_num} — {group['name']}:")
        for q_id, q_data in group["questions"]:
            field = q_data.get("required_field") or "(none)"
            question_text = q_data.get("question_text", q_data.get("description", ""))
            lines.append(f"  - {q_id}: \"{question_text}\"")
            lines.append(f"    Required field: {field}")
        lines.append("")
    return "\n".join(lines)


def _build_flat_description(intake_steps: dict) -> str:
    """Build flat description (old format, backward compat)."""
    lines = []
    for step_id, step in intake_steps.items():
        fields = ", ".join(step["required_fields"]) if step.get("required_fields") else "None"
        lines.append(f"- {step_id}: {step['name']}")
        lines.append(f"  Description: {step.get('description', '')}")
        lines.append(f"  Required fields: {fields}")
    return "\n".join(lines)


class CompletenessJudge(BaseJudge):
    """
    Evaluates whether Casey completes all intake questions.

    Analyzes the full transcript to determine:
    - Which questions were asked
    - Which information was captured
    - Which confirmations were given

    Returns per-question pass/fail with evidence and overall completion rate.
    """

    def __init__(self, llm_client=None, model: str = "gpt-4o", prompt_version_id: Optional[int] = None):
        # Load intake steps based on prompt version
        self.intake_steps = get_intake_steps(prompt_version_id)
        self._new_format = _is_new_format(self.intake_steps)
        steps_count = len(self.intake_steps)

        super().__init__(
            judge_id="completeness_intake",
            description=f"Evaluates completion of all {steps_count} intake questions",
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
            max_completion_tokens=16000,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError(f"LLM returned empty response (model={self.model})")
        return content

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
            steps_description=build_steps_description(self.intake_steps),
            transcript=context.get_full_transcript(),
        )

        try:
            # Use higher max_tokens for completeness evaluation
            response = self._call_llm_extended(
                system_prompt="You are evaluating intake process completeness. Be thorough and accurate in assessing each question. Return only valid JSON with no trailing commas.",
                user_prompt=prompt,
            )

            result = self._parse_completeness_response(response)
            step_results = result.get("step_results", {})

            # Calculate completion metrics
            questions_completed = 0
            questions_partial = 0
            questions_failed = 0
            questions_total = len(self.intake_steps)
            missing_fields = []

            for step_id, step_data in self.intake_steps.items():
                step_result = step_results.get(step_id, {})
                status = step_result.get("status", "fail")

                if status == "pass":
                    questions_completed += 1
                elif status == "partial":
                    questions_partial += 1
                    # Track missing field for partial questions (new format: singular field)
                    if self._new_format:
                        field = step_data.get("required_field")
                        if field:
                            captured = step_result.get("captured", "")
                            if not captured or field.lower() not in str(captured).lower():
                                missing_fields.append(field)
                    else:
                        for field in step_data.get("required_fields", []):
                            captured = step_result.get("captured", "")
                            if not captured or field.lower() not in str(captured).lower():
                                missing_fields.append(field)
                else:
                    questions_failed += 1
                    # All fields missing for failed questions
                    if self._new_format:
                        field = step_data.get("required_field")
                        if field:
                            missing_fields.append(field)
                    else:
                        missing_fields.extend(step_data.get("required_fields", []))

            # Calculate completion rate (partial counts as 0.5)
            completion_rate = (questions_completed + (questions_partial * 0.5)) / questions_total

            # Determine overall verdict
            if completion_rate >= 1.0:
                verdict = JudgeVerdict.PASS
            elif completion_rate >= 0.5:
                verdict = JudgeVerdict.PARTIAL
            else:
                verdict = JudgeVerdict.FAIL

            # Build step_results dict with fallback for missing entries
            final_step_results = {}
            for step_id in self.intake_steps:
                final_step_results[step_id] = step_results.get(step_id, {
                    "status": "fail",
                    "captured": None,
                    "evidence": "",
                    "reason": "Not found in evaluation",
                })
                # Attach step_data (name, parent info) for display
                final_step_results[step_id]["step_data"] = {
                    k: v for k, v in self.intake_steps[step_id].items()
                    if k in ("name", "parent_step", "parent_step_name")
                }

            return JudgeResult(
                judge_id=self.judge_id,
                verdict=verdict,
                score=round(completion_rate * 5, 1),  # Convert to 0-5 scale
                reasoning=result.get("reasoning", ""),
                evidence=[],
                metadata={
                    # New keys
                    "questions_completed": questions_completed,
                    "questions_partial": questions_partial,
                    "questions_failed": questions_failed,
                    "questions_total": questions_total,
                    # Old keys (backward compat)
                    "steps_completed": questions_completed,
                    "steps_partial": questions_partial,
                    "steps_failed": questions_failed,
                    "steps_total": questions_total,
                    "completion_rate": round(completion_rate, 2),
                    "step_results": final_step_results,
                    "missing_fields": list(set(missing_fields)),
                },
            )

        except Exception as e:
            return self._create_error_result(str(e))


class CompletenessEvaluator:
    """
    Coordinates completeness evaluation for intake conversations.
    """

    def __init__(self, llm_client=None, model: str = "gpt-4o", prompt_version_id: Optional[int] = None):
        self.judge = CompletenessJudge(llm_client, model, prompt_version_id)

    def evaluate(self, context: ConversationContext) -> JudgeResult:
        """Evaluate conversation for intake completeness."""
        return self.judge.evaluate(context)

    def get_summary(self, result: JudgeResult) -> dict:
        """Generate summary of completeness evaluation."""
        metadata = result.metadata or {}

        return {
            "questions_completed": metadata.get("questions_completed", metadata.get("steps_completed", 0)),
            "questions_partial": metadata.get("questions_partial", metadata.get("steps_partial", 0)),
            "questions_failed": metadata.get("questions_failed", metadata.get("steps_failed", 0)),
            "questions_total": metadata.get("questions_total", metadata.get("steps_total", 13)),
            "steps_completed": metadata.get("steps_completed", 0),
            "steps_partial": metadata.get("steps_partial", 0),
            "steps_failed": metadata.get("steps_failed", 0),
            "steps_total": metadata.get("steps_total", 13),
            "completion_rate": metadata.get("completion_rate", 0),
            "verdict": result.verdict.value,
            "missing_fields": metadata.get("missing_fields", []),
            "step_results": metadata.get("step_results", {}),
        }
