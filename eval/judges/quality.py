"""
Quality judges for evaluating conversation quality.

These judges assess subjective quality dimensions:
- Empathy
- Clarity
- Relevancy
- Language consistency
"""

from string import Template
from eval.judges.base import (
    BaseJudge,
    JudgeResult,
    JudgeVerdict,
    ConversationContext,
)


# =============================================================================
# EMPATHY JUDGE
# =============================================================================

EMPATHY_PROMPT = Template("""You are evaluating whether an AI intake agent responds with appropriate empathy.

CONTEXT: The client is seeking free legal aid, likely in a stressful situation. The agent should acknowledge their situation without being overly emotional or making promises.

CONVERSATION:
${transcript}

SCORING CRITERIA (1-5 scale):

1 = Cold/dismissive - Ignores emotional context, robotic responses, dismissive of concerns
2 = Neutral - Neither warm nor cold, purely transactional, misses obvious emotional cues
3 = Adequate - Basic acknowledgment of situation, appropriate but minimal empathy
4 = Warm - Good acknowledgment of concerns, uses client's name, validates feelings
5 = Excellent - Perfect balance of empathy and professionalism, naturally supportive

For each exchange where the client expresses concern, stress, or emotion, evaluate the agent's response.

Output a JSON object with:
- "score": Number 1-5
- "reasoning": 2-3 sentences explaining your score
- "positive_examples": List of quotes showing good empathy (if any)
- "negative_examples": List of quotes showing poor empathy (if any)

Return only valid JSON.""")


class EmpathyJudge(BaseJudge):
    """
    Evaluates empathy in agent responses.

    Scores on 1-5 scale for:
    - Acknowledgment of client's situation
    - Appropriate emotional response
    - Balance of warmth and professionalism
    """

    def __init__(self, llm_client=None, model: str = "gpt-4o"):
        super().__init__(
            judge_id="quality_empathy",
            description="Evaluates empathy and emotional intelligence in responses",
            llm_client=llm_client,
            model=model,
        )

    def evaluate(self, context: ConversationContext) -> JudgeResult:
        if self.llm_client is None:
            return self._create_error_result("LLM client required for empathy evaluation")

        prompt = EMPATHY_PROMPT.substitute(
            transcript=context.get_full_transcript()
        )

        try:
            response = self._call_llm(
                system_prompt="You are evaluating conversational empathy. Be balanced and fair in your assessment.",
                user_prompt=prompt,
            )

            result = self._parse_llm_json_response(response)
            score = result.get("score", 0)

            # Convert score to verdict
            if score >= 4:
                verdict = JudgeVerdict.PASS
            elif score >= 3:
                verdict = JudgeVerdict.PARTIAL
            else:
                verdict = JudgeVerdict.FAIL

            return JudgeResult(
                judge_id=self.judge_id,
                verdict=verdict,
                score=float(score),
                reasoning=result.get("reasoning", ""),
                evidence=result.get("positive_examples", []) + result.get("negative_examples", []),
                metadata={
                    "positive_examples": result.get("positive_examples", []),
                    "negative_examples": result.get("negative_examples", []),
                },
            )

        except Exception as e:
            return self._create_error_result(str(e))


# =============================================================================
# CLARITY JUDGE
# =============================================================================

CLARITY_PROMPT = Template("""You are evaluating whether an AI intake agent communicates clearly.

TARGET AUDIENCE: The agent is communicating with legal aid clients who may have:
- Limited English proficiency
- Various education levels
- High stress affecting comprehension
- No legal background

The target reading level is 8th grade.

CONVERSATION:
${transcript}

SCORING CRITERIA (1-5 scale):

1 = Very unclear - Complex sentences, unexplained jargon, confusing structure
2 = Somewhat unclear - Some jargon, overly long sentences, assumes knowledge
3 = Adequate - Generally clear but occasional complex language or unclear phrasing
4 = Clear - Simple language, good structure, explains terms when used
5 = Excellent - Very accessible, perfect for diverse audiences, no ambiguity

Evaluate the agent's messages for:
- Sentence complexity
- Use of jargon (legal, technical)
- Clarity of questions and instructions
- Overall accessibility

Output a JSON object with:
- "score": Number 1-5
- "reasoning": 2-3 sentences explaining your score
- "clear_examples": List of particularly clear statements
- "unclear_examples": List of confusing or complex statements

Return only valid JSON.""")


class ClarityJudge(BaseJudge):
    """
    Evaluates clarity of agent communications.

    Scores on 1-5 scale for:
    - Reading level appropriateness
    - Jargon usage
    - Sentence structure
    - Overall accessibility
    """

    def __init__(self, llm_client=None, model: str = "gpt-4o"):
        super().__init__(
            judge_id="quality_clarity",
            description="Evaluates clarity and accessibility of communications",
            llm_client=llm_client,
            model=model,
        )

    def evaluate(self, context: ConversationContext) -> JudgeResult:
        if self.llm_client is None:
            return self._create_error_result("LLM client required for clarity evaluation")

        prompt = CLARITY_PROMPT.substitute(
            transcript=context.get_full_transcript()
        )

        try:
            response = self._call_llm(
                system_prompt="You are evaluating communication clarity for diverse audiences.",
                user_prompt=prompt,
            )

            result = self._parse_llm_json_response(response)
            score = result.get("score", 0)

            if score >= 4:
                verdict = JudgeVerdict.PASS
            elif score >= 3:
                verdict = JudgeVerdict.PARTIAL
            else:
                verdict = JudgeVerdict.FAIL

            return JudgeResult(
                judge_id=self.judge_id,
                verdict=verdict,
                score=float(score),
                reasoning=result.get("reasoning", ""),
                evidence=result.get("unclear_examples", []),
                metadata={
                    "clear_examples": result.get("clear_examples", []),
                    "unclear_examples": result.get("unclear_examples", []),
                },
            )

        except Exception as e:
            return self._create_error_result(str(e))


# =============================================================================
# RELEVANCY JUDGE
# =============================================================================

RELEVANCY_PROMPT = Template("""You are evaluating whether an AI intake agent's responses are relevant to what the client said.

CONVERSATION:
${transcript}

SCORING CRITERIA (1-5 scale):

1 = Irrelevant - Responses don't address what client said, ignores questions
2 = Somewhat relevant - Partially addresses client but misses key points
3 = Adequate - Addresses main point but may miss nuances or secondary questions
4 = Relevant - Directly addresses what client said, appropriate follow-ups
5 = Excellent - Perfectly addresses all aspects of client's message, anticipates needs

For each client message, evaluate whether the agent's response:
- Addresses the actual content of what was said
- Responds to any questions asked
- Acknowledges any concerns raised
- Stays on topic while advancing the intake

Output a JSON object with:
- "score": Number 1-5
- "reasoning": 2-3 sentences explaining your score
- "relevant_exchanges": List of well-handled exchanges
- "irrelevant_exchanges": List of poorly handled exchanges

Return only valid JSON.""")


class RelevancyJudge(BaseJudge):
    """
    Evaluates relevancy of agent responses to client messages.

    Scores on 1-5 scale for:
    - Addressing client's actual message
    - Responding to questions
    - Acknowledging concerns
    - Staying on topic
    """

    def __init__(self, llm_client=None, model: str = "gpt-4o"):
        super().__init__(
            judge_id="quality_relevancy",
            description="Evaluates relevancy of responses to client messages",
            llm_client=llm_client,
            model=model,
        )

    def evaluate(self, context: ConversationContext) -> JudgeResult:
        if self.llm_client is None:
            return self._create_error_result("LLM client required for relevancy evaluation")

        prompt = RELEVANCY_PROMPT.substitute(
            transcript=context.get_full_transcript()
        )

        try:
            response = self._call_llm(
                system_prompt="You are evaluating conversational relevancy.",
                user_prompt=prompt,
            )

            result = self._parse_llm_json_response(response)
            score = result.get("score", 0)

            if score >= 4:
                verdict = JudgeVerdict.PASS
            elif score >= 3:
                verdict = JudgeVerdict.PARTIAL
            else:
                verdict = JudgeVerdict.FAIL

            return JudgeResult(
                judge_id=self.judge_id,
                verdict=verdict,
                score=float(score),
                reasoning=result.get("reasoning", ""),
                evidence=result.get("irrelevant_exchanges", []),
                metadata={
                    "relevant_exchanges": result.get("relevant_exchanges", []),
                    "irrelevant_exchanges": result.get("irrelevant_exchanges", []),
                },
            )

        except Exception as e:
            return self._create_error_result(str(e))


# =============================================================================
# LANGUAGE CONSISTENCY JUDGE
# =============================================================================

LANGUAGE_CONSISTENCY_PROMPT = Template("""You are evaluating whether an AI intake agent maintains consistent language use.

The client selected: ${selected_language}

CONVERSATION:
${transcript}

EVALUATION CRITERIA:

1. After language selection, all agent responses should be in the selected language
2. If the client switches languages, the agent should acknowledge and confirm preference
3. Agent should NOT randomly switch to English mid-conversation
4. Questions, instructions, and all content should be in the selected language

Output a JSON object with:
- "verdict": "pass", "fail", or "not_applicable" (if English was selected)
- "reasoning": 2-3 sentences explaining your evaluation
- "language_switches": List of any instances where agent switched languages inappropriately
- "messages_in_wrong_language": Count of agent messages in wrong language

Return only valid JSON.""")


class LanguageConsistencyJudge(BaseJudge):
    """
    Evaluates whether agent maintains selected language throughout conversation.

    Pass/Fail evaluation for:
    - Consistent language use
    - Appropriate handling of language switches
    - No random English insertions
    """

    def __init__(self, llm_client=None, model: str = "gpt-4o"):
        super().__init__(
            judge_id="quality_language_consistency",
            description="Evaluates consistent language use throughout conversation",
            llm_client=llm_client,
            model=model,
        )

    def evaluate(self, context: ConversationContext) -> JudgeResult:
        if self.llm_client is None:
            return self._create_error_result("LLM client required for language evaluation")

        # Determine selected language from persona or saved data
        selected_language = "English"
        if context.persona:
            selected_language = context.persona.get("primary_language", "English")
        if context.saved_data and "q1" in context.saved_data:
            selected_language = context.saved_data["q1"]

        # If English, this check is not applicable
        if selected_language.lower() == "english":
            return self._create_not_applicable_result("English selected, language consistency N/A")

        prompt = LANGUAGE_CONSISTENCY_PROMPT.substitute(
            selected_language=selected_language,
            transcript=context.get_full_transcript(),
        )

        try:
            response = self._call_llm(
                system_prompt="You are evaluating multilingual conversation consistency.",
                user_prompt=prompt,
            )

            result = self._parse_llm_json_response(response)

            verdict_str = result.get("verdict", "error").lower()
            verdict_map = {
                "pass": JudgeVerdict.PASS,
                "fail": JudgeVerdict.FAIL,
                "not_applicable": JudgeVerdict.NOT_APPLICABLE,
            }
            verdict = verdict_map.get(verdict_str, JudgeVerdict.ERROR)

            return JudgeResult(
                judge_id=self.judge_id,
                verdict=verdict,
                reasoning=result.get("reasoning", ""),
                evidence=result.get("language_switches", []),
                metadata={
                    "selected_language": selected_language,
                    "messages_in_wrong_language": result.get("messages_in_wrong_language", 0),
                },
            )

        except Exception as e:
            return self._create_error_result(str(e))


# =============================================================================
# QUALITY EVALUATOR
# =============================================================================

class QualityEvaluator:
    """
    Coordinates all quality judges for comprehensive quality evaluation.
    """

    def __init__(self, llm_client=None, model: str = "gpt-4o"):
        self.judges = [
            EmpathyJudge(llm_client, model),
            ClarityJudge(llm_client, model),
            RelevancyJudge(llm_client, model),
            LanguageConsistencyJudge(llm_client, model),
        ]

    def evaluate_all(self, context: ConversationContext) -> list[JudgeResult]:
        """Evaluate conversation against all quality criteria."""
        return [judge.evaluate(context) for judge in self.judges]

    def get_summary(self, results: list[JudgeResult]) -> dict:
        """Generate summary of quality evaluation results."""
        scored_results = [r for r in results if r.score is not None]

        summary = {
            "total": len(results),
            "average_score": sum(r.score for r in scored_results) / len(scored_results) if scored_results else 0,
            "scores": {r.judge_id: r.score for r in results if r.score is not None},
            "verdicts": {r.judge_id: r.verdict.value for r in results},
            "passed": sum(1 for r in results if r.verdict == JudgeVerdict.PASS),
            "failed": sum(1 for r in results if r.verdict == JudgeVerdict.FAIL),
            "partial": sum(1 for r in results if r.verdict == JudgeVerdict.PARTIAL),
        }

        return summary
