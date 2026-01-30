"""
Main evaluation runner that orchestrates the full evaluation pipeline.

Coordinates:
1. Persona generation
2. Conversation simulation via HTTP API
3. Evaluation with judges
4. Results aggregation
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import datetime
import json

from eval.personas.models import Persona
from eval.personas.generator import PersonaGenerator
from eval.personas.edge_cases import ALL_EDGE_CASE_PERSONAS, SAFETY_PERSONAS
from eval.simulation.conversation import (
    ConversationRunner,
    AgentforceConversationRunner,
    ConversationResult,
    CaseyAPIConfig,
    AgentforceConfig,
)
from eval.judges.base import JudgeResult, JudgeVerdict, ConversationContext
from eval.judges.behavioral import BehavioralRuleEvaluator
from eval.judges.safety import SafetyEvaluator
from eval.judges.quality import QualityEvaluator


@dataclass
class EvaluationConfig:
    """Configuration for an evaluation run."""

    # API configuration - supports both legacy and Agentforce
    casey_api_url: str  # For legacy: base URL. For Agentforce: my_domain_url
    casey_api_key: Optional[str] = None  # For legacy: API key. For Agentforce: access_token

    # Agentforce-specific configuration
    agentforce_agent_id: Optional[str] = None  # 18-char agent ID (enables Agentforce mode)
    use_agentforce: bool = False  # Explicitly use Agentforce API

    # Conversation settings
    max_turns_per_conversation: int = 100
    turn_delay: float = 0.5

    # Persona settings
    num_random_personas: int = 10
    include_edge_cases: bool = True
    include_safety_personas: bool = True

    # Judge settings
    run_behavioral_judges: bool = True
    run_safety_judges: bool = True
    run_quality_judges: bool = True

    # LLM settings
    judge_model: str = "gpt-4o"
    synthetic_client_model: str = "gpt-4.1-mini"

    def is_agentforce(self) -> bool:
        """Check if Agentforce API should be used."""
        return self.use_agentforce or self.agentforce_agent_id is not None


@dataclass
class ConversationEvaluation:
    """Evaluation results for a single conversation."""

    conversation: ConversationResult
    behavioral_results: list[JudgeResult] = field(default_factory=list)
    safety_results: list[JudgeResult] = field(default_factory=list)
    quality_results: list[JudgeResult] = field(default_factory=list)

    def get_all_results(self) -> list[JudgeResult]:
        return self.behavioral_results + self.safety_results + self.quality_results

    def passed_all_safety(self) -> bool:
        """Check if all applicable safety checks passed."""
        for result in self.safety_results:
            if result.verdict == JudgeVerdict.FAIL:
                return False
        return True

    def get_quality_scores(self) -> dict[str, float]:
        """Get quality scores by judge."""
        return {
            r.judge_id: r.score
            for r in self.quality_results
            if r.score is not None
        }

    def to_dict(self) -> dict:
        return {
            "thread_id": self.conversation.thread_id,
            "persona": self.conversation.persona.name,
            "completed": self.conversation.completed,
            "completion_reason": self.conversation.completion_reason,
            "turn_count": self.conversation.turn_count,
            "duration_seconds": self.conversation.duration_seconds,
            "behavioral_results": [r.to_dict() for r in self.behavioral_results],
            "safety_results": [r.to_dict() for r in self.safety_results],
            "quality_results": [r.to_dict() for r in self.quality_results],
        }


@dataclass
class EvaluationSummary:
    """Summary of a full evaluation run."""

    run_id: str
    timestamp: str
    config: EvaluationConfig
    total_conversations: int
    completed_conversations: int
    failed_conversations: int

    # Behavioral metrics
    behavioral_pass_rate: float
    behavioral_by_category: dict

    # Safety metrics
    safety_pass_rate: float
    safety_failures: list[dict]

    # Quality metrics
    average_quality_scores: dict[str, float]

    # Individual evaluations
    evaluations: list[ConversationEvaluation]

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "total_conversations": self.total_conversations,
            "completed_conversations": self.completed_conversations,
            "failed_conversations": self.failed_conversations,
            "behavioral_pass_rate": self.behavioral_pass_rate,
            "behavioral_by_category": self.behavioral_by_category,
            "safety_pass_rate": self.safety_pass_rate,
            "safety_failures": self.safety_failures,
            "average_quality_scores": self.average_quality_scores,
            "evaluations": [e.to_dict() for e in self.evaluations],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class EvaluationRunner:
    """
    Main evaluation runner that orchestrates the full pipeline.

    Usage:
        runner = EvaluationRunner(config, openai_client)
        summary = runner.run()
        print(summary.to_json())
    """

    def __init__(
        self,
        config: EvaluationConfig,
        llm_client,  # OpenAI client
    ):
        """
        Initialize the evaluation runner.

        Args:
            config: Evaluation configuration
            llm_client: OpenAI client for judges and synthetic client
        """
        self.config = config
        self.llm_client = llm_client

        # Initialize conversation runner based on API type
        if config.is_agentforce():
            # Use Salesforce Agentforce API
            if not config.agentforce_agent_id:
                raise ValueError("agentforce_agent_id required when using Agentforce API")
            if not config.casey_api_key:
                raise ValueError("casey_api_key (access_token) required for Agentforce API")

            agentforce_config = AgentforceConfig(
                my_domain_url=config.casey_api_url,
                agent_id=config.agentforce_agent_id,
                access_token=config.casey_api_key,
            )
            self.conversation_runner = AgentforceConversationRunner(
                config=agentforce_config,
                llm_client=llm_client,
                synthetic_model=config.synthetic_client_model,
            )
        else:
            # Use legacy/generic HTTP API
            api_config = CaseyAPIConfig(
                base_url=config.casey_api_url,
                api_key=config.casey_api_key,
            )
            self.conversation_runner = ConversationRunner(
                api_config=api_config,
                llm_client=llm_client,
                synthetic_model=config.synthetic_client_model,
            )

        self.persona_generator = PersonaGenerator()

        # Initialize judges
        self.behavioral_evaluator = BehavioralRuleEvaluator(
            llm_client=llm_client,
            model=config.judge_model,
        ) if config.run_behavioral_judges else None

        self.safety_evaluator = SafetyEvaluator(
            llm_client=llm_client,
            model=config.judge_model,
        ) if config.run_safety_judges else None

        self.quality_evaluator = QualityEvaluator(
            llm_client=llm_client,
            model=config.judge_model,
        ) if config.run_quality_judges else None

    def _generate_personas(self) -> list[Persona]:
        """Generate personas for evaluation."""
        personas = []

        # Add edge case personas
        if self.config.include_edge_cases:
            personas.extend(ALL_EDGE_CASE_PERSONAS)
        elif self.config.include_safety_personas:
            personas.extend(SAFETY_PERSONAS)

        # Add random personas
        if self.config.num_random_personas > 0:
            random_personas = self.persona_generator.generate_batch(
                count=self.config.num_random_personas,
                include_edge_cases=True,
            )
            personas.extend(random_personas)

        return personas

    def _evaluate_conversation(
        self,
        conversation: ConversationResult,
    ) -> ConversationEvaluation:
        """Evaluate a single conversation with all judges."""
        context = conversation.to_context()

        evaluation = ConversationEvaluation(conversation=conversation)

        # Run behavioral judges
        if self.behavioral_evaluator:
            evaluation.behavioral_results = self.behavioral_evaluator.evaluate_all(context)

        # Run safety judges
        if self.safety_evaluator:
            evaluation.safety_results = self.safety_evaluator.evaluate_all(context)

        # Run quality judges
        if self.quality_evaluator:
            evaluation.quality_results = self.quality_evaluator.evaluate_all(context)

        return evaluation

    def _compute_summary(
        self,
        evaluations: list[ConversationEvaluation],
        run_id: str,
    ) -> EvaluationSummary:
        """Compute summary statistics from evaluations."""

        # Conversation stats
        total = len(evaluations)
        completed = sum(1 for e in evaluations if e.conversation.completed)
        failed = total - completed

        # Behavioral stats
        all_behavioral = [r for e in evaluations for r in e.behavioral_results]
        applicable_behavioral = [
            r for r in all_behavioral
            if r.verdict not in [JudgeVerdict.NOT_APPLICABLE, JudgeVerdict.ERROR]
        ]
        passed_behavioral = sum(1 for r in applicable_behavioral if r.verdict == JudgeVerdict.PASS)
        behavioral_pass_rate = passed_behavioral / len(applicable_behavioral) if applicable_behavioral else 0

        # Behavioral by category
        behavioral_by_category = {}
        for result in all_behavioral:
            cat = result.metadata.get("rule_category", "unknown")
            if cat not in behavioral_by_category:
                behavioral_by_category[cat] = {"passed": 0, "failed": 0, "total": 0}
            behavioral_by_category[cat]["total"] += 1
            if result.verdict == JudgeVerdict.PASS:
                behavioral_by_category[cat]["passed"] += 1
            elif result.verdict == JudgeVerdict.FAIL:
                behavioral_by_category[cat]["failed"] += 1

        # Safety stats
        all_safety = [r for e in evaluations for r in e.safety_results]
        applicable_safety = [
            r for r in all_safety
            if r.verdict not in [JudgeVerdict.NOT_APPLICABLE, JudgeVerdict.ERROR]
        ]
        passed_safety = sum(1 for r in applicable_safety if r.verdict == JudgeVerdict.PASS)
        safety_pass_rate = passed_safety / len(applicable_safety) if applicable_safety else 1.0

        # Collect safety failures
        safety_failures = []
        for e in evaluations:
            for r in e.safety_results:
                if r.verdict == JudgeVerdict.FAIL:
                    safety_failures.append({
                        "thread_id": e.conversation.thread_id,
                        "persona": e.conversation.persona.name,
                        "judge": r.judge_id,
                        "reasoning": r.reasoning,
                    })

        # Quality stats
        all_quality = [r for e in evaluations for r in e.quality_results]
        quality_by_judge = {}
        for result in all_quality:
            if result.score is not None:
                if result.judge_id not in quality_by_judge:
                    quality_by_judge[result.judge_id] = []
                quality_by_judge[result.judge_id].append(result.score)

        avg_quality = {
            judge_id: sum(scores) / len(scores)
            for judge_id, scores in quality_by_judge.items()
        }

        return EvaluationSummary(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            config=self.config,
            total_conversations=total,
            completed_conversations=completed,
            failed_conversations=failed,
            behavioral_pass_rate=behavioral_pass_rate,
            behavioral_by_category=behavioral_by_category,
            safety_pass_rate=safety_pass_rate,
            safety_failures=safety_failures,
            average_quality_scores=avg_quality,
            evaluations=evaluations,
        )

    def run(
        self,
        run_id: Optional[str] = None,
        on_conversation: Optional[Callable[[int, ConversationResult], None]] = None,
        on_evaluation: Optional[Callable[[int, ConversationEvaluation], None]] = None,
    ) -> EvaluationSummary:
        """
        Run the full evaluation pipeline.

        Args:
            run_id: Optional identifier for this run
            on_conversation: Callback after each conversation completes
            on_evaluation: Callback after each evaluation completes

        Returns:
            EvaluationSummary with all results
        """
        if run_id is None:
            run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Generate personas
        personas = self._generate_personas()

        evaluations = []

        for i, persona in enumerate(personas):
            # Run conversation
            conversation = self.conversation_runner.run(
                persona=persona,
                max_turns=self.config.max_turns_per_conversation,
                turn_delay=self.config.turn_delay,
            )

            if on_conversation:
                on_conversation(i, conversation)

            # Evaluate conversation
            evaluation = self._evaluate_conversation(conversation)
            evaluations.append(evaluation)

            if on_evaluation:
                on_evaluation(i, evaluation)

        # Compute summary
        summary = self._compute_summary(evaluations, run_id)

        return summary

    def run_single(
        self,
        persona: Persona,
    ) -> ConversationEvaluation:
        """
        Run evaluation for a single persona.

        Args:
            persona: The persona to evaluate

        Returns:
            ConversationEvaluation for this persona
        """
        conversation = self.conversation_runner.run(
            persona=persona,
            max_turns=self.config.max_turns_per_conversation,
            turn_delay=self.config.turn_delay,
        )

        return self._evaluate_conversation(conversation)

    def close(self):
        """Clean up resources."""
        self.conversation_runner.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
