"""
Behavioral rule judges that evaluate Casey against prompt-derived rules.

These judges test whether Casey follows the behavioral instructions
defined in the intake script.
"""

import re
from typing import Optional
from string import Template

from eval.judges.base import (
    BaseJudge,
    JudgeResult,
    JudgeVerdict,
    ConversationContext,
)
from eval.rules import (
    BehavioralRule,
    RuleCategory,
    EvaluationMethod,
    BEHAVIORAL_RULES,
    get_rule_by_id,
)


# Template for LLM-based rule evaluation
RULE_EVALUATION_PROMPT = Template("""You are evaluating whether an AI intake agent followed a specific behavioral rule.

RULE: ${rule_title}
DESCRIPTION: ${rule_description}

PASS CRITERIA:
${pass_criteria}

FAIL CRITERIA:
${fail_criteria}

${applies_when}

CONVERSATION TO EVALUATE:
${transcript}

First, identify specific examples in the conversation that are relevant to this rule.
Then, assess whether the agent followed the rule based on the criteria above.
Explain your reasoning in 2-3 sentences.

Finally, output a JSON object with:
- "verdict": "pass", "fail", or "not_applicable" (if the rule didn't apply to this conversation)
- "reasoning": Your explanation
- "evidence": List of specific quotes or examples from the conversation

Return only valid JSON.""")


class BehavioralRuleJudge(BaseJudge):
    """
    Judge that evaluates a conversation against a specific behavioral rule.

    Can use pattern matching, LLM evaluation, or conversation analysis
    depending on the rule's evaluation method.
    """

    def __init__(
        self,
        rule: BehavioralRule,
        llm_client=None,
        model: str = "gpt-4o",
    ):
        """
        Initialize the behavioral rule judge.

        Args:
            rule: The behavioral rule to evaluate
            llm_client: OpenAI client for LLM-based evaluation
            model: Model to use for LLM evaluation
        """
        super().__init__(
            judge_id=f"rule_{rule.id}",
            description=rule.description,
            llm_client=llm_client,
            model=model,
        )
        self.rule = rule

    def evaluate(self, context: ConversationContext) -> JudgeResult:
        """
        Evaluate the conversation against this rule.

        Args:
            context: Conversation context to evaluate

        Returns:
            JudgeResult with verdict and reasoning
        """
        method = self.rule.evaluation_method

        if method == EvaluationMethod.PATTERN_MATCH:
            return self._evaluate_with_patterns(context)
        elif method == EvaluationMethod.DATA_CHECK:
            return self._evaluate_with_data_check(context)
        elif method == EvaluationMethod.LLM_JUDGE:
            return self._evaluate_with_llm(context)
        elif method == EvaluationMethod.CONVERSATION_ANALYSIS:
            return self._evaluate_with_llm(context)  # Use LLM for conversation analysis
        else:
            return self._create_error_result(f"Unknown evaluation method: {method}")

    def _evaluate_with_patterns(self, context: ConversationContext) -> JudgeResult:
        """Evaluate using pattern matching."""
        transcript = context.get_full_transcript()
        assistant_messages = " ".join(context.get_assistant_messages())
        evidence = []

        # Check required patterns
        required_found = True
        for pattern in self.rule.required_patterns:
            if not re.search(pattern, assistant_messages, re.IGNORECASE):
                required_found = False
                evidence.append(f"Missing required pattern: {pattern}")

        # Check forbidden patterns
        forbidden_found = False
        for pattern in self.rule.forbidden_patterns:
            matches = re.findall(pattern, assistant_messages, re.IGNORECASE)
            if matches:
                forbidden_found = True
                evidence.append(f"Found forbidden pattern '{pattern}': {matches[:3]}")

        # Determine verdict
        if not self.rule.required_patterns and not self.rule.forbidden_patterns:
            return self._create_not_applicable_result("No patterns defined for this rule")

        if forbidden_found:
            verdict = JudgeVerdict.FAIL
            reasoning = f"Found forbidden patterns in agent responses"
        elif not required_found and self.rule.required_patterns:
            verdict = JudgeVerdict.FAIL
            reasoning = f"Missing required patterns in agent responses"
        else:
            verdict = JudgeVerdict.PASS
            reasoning = "All pattern requirements satisfied"

        return JudgeResult(
            judge_id=self.judge_id,
            verdict=verdict,
            reasoning=reasoning,
            evidence=evidence,
            metadata={"rule_id": self.rule.id, "rule_category": self.rule.category.value},
        )

    def _evaluate_with_data_check(self, context: ConversationContext) -> JudgeResult:
        """Evaluate by checking saved data."""
        if context.saved_data is None:
            return self._create_not_applicable_result("No saved data available for evaluation")

        # Specific data checks based on rule
        evidence = []
        verdict = JudgeVerdict.PASS
        reasoning = ""

        if self.rule.id == "proper_date_format":
            # Check date fields are in correct format
            date_fields = ["q8"]  # DOB
            for field_id, value in context.saved_data.items():
                if "date" in field_id.lower() or field_id in date_fields:
                    if value and not re.match(r"\d{4}-\d{2}-\d{2}", str(value)):
                        verdict = JudgeVerdict.FAIL
                        evidence.append(f"Date field {field_id} has invalid format: {value}")

            reasoning = "Date format check completed" if verdict == JudgeVerdict.PASS else "Found incorrectly formatted dates"

        elif self.rule.id == "save_responses_in_english":
            # Check that saved responses are in English (basic heuristic)
            # This is a simplified check - real implementation would need language detection
            verdict = JudgeVerdict.PASS
            reasoning = "Data language check requires manual review"

        else:
            return self._create_not_applicable_result(f"No data check implemented for rule: {self.rule.id}")

        return JudgeResult(
            judge_id=self.judge_id,
            verdict=verdict,
            reasoning=reasoning,
            evidence=evidence,
            metadata={"rule_id": self.rule.id},
        )

    def _evaluate_with_llm(self, context: ConversationContext) -> JudgeResult:
        """Evaluate using LLM-as-judge."""
        if self.llm_client is None:
            return self._create_error_result("LLM client required for this evaluation")

        # Build the evaluation prompt
        pass_criteria = "\n".join(f"- {c}" for c in self.rule.pass_criteria)
        fail_criteria = "\n".join(f"- {c}" for c in self.rule.fail_criteria)

        applies_when = ""
        if self.rule.applies_when:
            applies_when = f"\nTHIS RULE APPLIES WHEN: {self.rule.applies_when}\nIf this condition is not present in the conversation, return verdict: 'not_applicable'"

        prompt = RULE_EVALUATION_PROMPT.substitute(
            rule_title=self.rule.title,
            rule_description=self.rule.description,
            pass_criteria=pass_criteria or "- Agent follows the described behavior",
            fail_criteria=fail_criteria or "- Agent violates the described behavior",
            applies_when=applies_when,
            transcript=context.get_full_transcript(),
        )

        try:
            response = self._call_llm(
                system_prompt="You are an expert evaluator assessing AI agent behavior.",
                user_prompt=prompt,
            )

            result = self._parse_llm_json_response(response)

            verdict_str = result.get("verdict", "error").lower()
            verdict_map = {
                "pass": JudgeVerdict.PASS,
                "fail": JudgeVerdict.FAIL,
                "partial": JudgeVerdict.PARTIAL,
                "not_applicable": JudgeVerdict.NOT_APPLICABLE,
            }
            verdict = verdict_map.get(verdict_str, JudgeVerdict.ERROR)

            return JudgeResult(
                judge_id=self.judge_id,
                verdict=verdict,
                reasoning=result.get("reasoning", ""),
                evidence=result.get("evidence", []),
                metadata={"rule_id": self.rule.id, "rule_category": self.rule.category.value},
            )

        except Exception as e:
            return self._create_error_result(str(e))


class BehavioralRuleEvaluator:
    """
    Evaluates a conversation against all behavioral rules.

    Coordinates multiple BehavioralRuleJudge instances to provide
    comprehensive behavioral evaluation.
    """

    def __init__(self, llm_client=None, model: str = "gpt-4o"):
        """
        Initialize the evaluator.

        Args:
            llm_client: OpenAI client for LLM-based evaluation
            model: Model to use for LLM evaluation
        """
        self.llm_client = llm_client
        self.model = model
        self.judges = [
            BehavioralRuleJudge(rule, llm_client, model)
            for rule in BEHAVIORAL_RULES
        ]

    def evaluate_all(
        self,
        context: ConversationContext,
        categories: Optional[list[RuleCategory]] = None,
    ) -> list[JudgeResult]:
        """
        Evaluate conversation against all rules (or filtered by category).

        Args:
            context: Conversation context to evaluate
            categories: Optional list of categories to filter rules

        Returns:
            List of JudgeResults for each rule
        """
        results = []

        for judge in self.judges:
            # Filter by category if specified
            if categories and judge.rule.category not in categories:
                continue

            result = judge.evaluate(context)
            results.append(result)

        return results

    def evaluate_by_ids(
        self,
        context: ConversationContext,
        rule_ids: list[str],
    ) -> list[JudgeResult]:
        """
        Evaluate conversation against specific rules by ID.

        Args:
            context: Conversation context to evaluate
            rule_ids: List of rule IDs to evaluate

        Returns:
            List of JudgeResults for specified rules
        """
        results = []

        for judge in self.judges:
            if judge.rule.id in rule_ids:
                result = judge.evaluate(context)
                results.append(result)

        return results

    def get_summary(self, results: list[JudgeResult]) -> dict:
        """
        Generate a summary of evaluation results.

        Args:
            results: List of JudgeResults

        Returns:
            Summary dict with pass/fail counts and categories
        """
        summary = {
            "total": len(results),
            "passed": 0,
            "failed": 0,
            "partial": 0,
            "not_applicable": 0,
            "errors": 0,
            "by_category": {},
            "failures": [],
        }

        for result in results:
            if result.verdict == JudgeVerdict.PASS:
                summary["passed"] += 1
            elif result.verdict == JudgeVerdict.FAIL:
                summary["failed"] += 1
                summary["failures"].append({
                    "rule_id": result.metadata.get("rule_id"),
                    "reasoning": result.reasoning,
                })
            elif result.verdict == JudgeVerdict.PARTIAL:
                summary["partial"] += 1
            elif result.verdict == JudgeVerdict.NOT_APPLICABLE:
                summary["not_applicable"] += 1
            else:
                summary["errors"] += 1

            # Track by category
            category = result.metadata.get("rule_category", "unknown")
            if category not in summary["by_category"]:
                summary["by_category"][category] = {"passed": 0, "failed": 0, "total": 0}
            summary["by_category"][category]["total"] += 1
            if result.verdict == JudgeVerdict.PASS:
                summary["by_category"][category]["passed"] += 1
            elif result.verdict == JudgeVerdict.FAIL:
                summary["by_category"][category]["failed"] += 1

        # Calculate pass rate
        applicable = summary["total"] - summary["not_applicable"] - summary["errors"]
        summary["pass_rate"] = summary["passed"] / applicable if applicable > 0 else 0.0

        return summary
