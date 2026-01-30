"""
Base class for LLM-as-judge evaluators.

Provides common functionality for all judge implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import json


class JudgeVerdict(Enum):
    """Possible verdicts from a judge."""
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"


@dataclass
class JudgeResult:
    """Result from a judge evaluation."""

    judge_id: str
    verdict: JudgeVerdict
    score: Optional[float] = None  # 0-5 scale for quality metrics
    reasoning: str = ""
    evidence: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "judge_id": self.judge_id,
            "verdict": self.verdict.value,
            "score": self.score,
            "reasoning": self.reasoning,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "JudgeResult":
        return cls(
            judge_id=data["judge_id"],
            verdict=JudgeVerdict(data["verdict"]),
            score=data.get("score"),
            reasoning=data.get("reasoning", ""),
            evidence=data.get("evidence", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Context for evaluating a conversation."""

    turns: list[ConversationTurn]
    persona: Optional[dict] = None
    saved_data: Optional[dict] = None
    salesforce_data: Optional[dict] = None
    tool_calls: list[dict] = field(default_factory=list)

    def get_full_transcript(self) -> str:
        """Get the full conversation as a formatted string."""
        lines = []
        for turn in self.turns:
            prefix = "Client" if turn.role == "user" else "Casey"
            lines.append(f"{prefix}: {turn.content}")
        return "\n\n".join(lines)

    def get_assistant_messages(self) -> list[str]:
        """Get only assistant/Casey messages."""
        return [t.content for t in self.turns if t.role == "assistant"]

    def get_user_messages(self) -> list[str]:
        """Get only user/client messages."""
        return [t.content for t in self.turns if t.role == "user"]

    def get_exchanges(self) -> list[tuple[str, str]]:
        """Get paired exchanges (user message, assistant response)."""
        exchanges = []
        for i in range(0, len(self.turns) - 1, 2):
            if i + 1 < len(self.turns):
                user_msg = self.turns[i].content if self.turns[i].role == "user" else ""
                asst_msg = self.turns[i + 1].content if self.turns[i + 1].role == "assistant" else ""
                if user_msg or asst_msg:
                    exchanges.append((user_msg, asst_msg))
        return exchanges


class BaseJudge(ABC):
    """
    Base class for all LLM-as-judge evaluators.

    Judges evaluate conversations against specific criteria using
    either pattern matching, LLM evaluation, or data checks.
    """

    def __init__(
        self,
        judge_id: str,
        description: str,
        llm_client=None,
        model: str = "gpt-4o",
    ):
        """
        Initialize the judge.

        Args:
            judge_id: Unique identifier for this judge
            description: Human-readable description of what this judges
            llm_client: OpenAI/Anthropic client for LLM-based evaluation
            model: Model to use for LLM evaluation
        """
        self.judge_id = judge_id
        self.description = description
        self.llm_client = llm_client
        self.model = model

    @abstractmethod
    def evaluate(self, context: ConversationContext) -> JudgeResult:
        """
        Evaluate a conversation against this judge's criteria.

        Args:
            context: The conversation context to evaluate

        Returns:
            JudgeResult with verdict, score, and reasoning
        """
        pass

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the LLM for evaluation.

        Args:
            system_prompt: System prompt for the judge
            user_prompt: User prompt with content to evaluate

        Returns:
            LLM response text
        """
        if self.llm_client is None:
            raise ValueError("LLM client required for this evaluation")

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent evaluation
            max_tokens=1024,
        )

        return response.choices[0].message.content

    def _parse_llm_json_response(self, response: str) -> dict:
        """
        Parse JSON from LLM response, handling common formatting issues.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed JSON dict
        """
        # Try to extract JSON from response
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

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            raise

    def _create_error_result(self, error_msg: str) -> JudgeResult:
        """Create an error result."""
        return JudgeResult(
            judge_id=self.judge_id,
            verdict=JudgeVerdict.ERROR,
            reasoning=f"Evaluation error: {error_msg}",
        )

    def _create_not_applicable_result(self, reason: str) -> JudgeResult:
        """Create a not-applicable result."""
        return JudgeResult(
            judge_id=self.judge_id,
            verdict=JudgeVerdict.NOT_APPLICABLE,
            reasoning=reason,
        )
