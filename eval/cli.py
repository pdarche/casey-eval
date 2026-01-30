"""
CLI interface for the Casey evaluation suite.

Usage:
    python -m eval.cli run-eval --conversations 10 --output results.json
    python -m eval.cli run-single --persona "CRISIS_DV_DISCLOSURE" --preview
    python -m eval.cli list-personas
    python -m eval.cli list-rules
"""

import click
import json
import os
from pathlib import Path
from datetime import datetime

# Conditional imports - these require dependencies
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@click.group()
def cli():
    """Casey Evaluation Suite CLI."""
    pass


@cli.command()
@click.option("--conversations", "-n", default=10, help="Number of conversations to run")
@click.option("--output", "-o", default=None, help="Output file for results (JSON)")
@click.option("--api-url", envvar="CASEY_API_URL", required=True, help="Salesforce My Domain URL or legacy API URL")
@click.option("--api-key", envvar="CASEY_API_KEY", default=None, help="Access token (Agentforce) or API key (legacy)")
@click.option("--agent-id", envvar="AGENTFORCE_AGENT_ID", default=None, help="Agentforce 18-char agent ID")
@click.option("--include-edge-cases/--no-edge-cases", default=True, help="Include edge case personas")
@click.option("--judge-model", default="gpt-4o", help="Model for LLM judges")
@click.option("--client-model", default="gpt-4.1-mini", help="Model for synthetic client")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def run_eval(
    conversations: int,
    output: str,
    api_url: str,
    api_key: str,
    agent_id: str,
    include_edge_cases: bool,
    judge_model: str,
    client_model: str,
    verbose: bool,
):
    """Run full evaluation suite.

    For Salesforce Agentforce, provide:
      --api-url: Your Salesforce My Domain URL (e.g., https://your-org.my.salesforce.com)
      --api-key: OAuth access token from connected app
      --agent-id: 18-character Agent ID from Agent Overview page

    Environment variables: CASEY_API_URL, CASEY_API_KEY, AGENTFORCE_AGENT_ID
    """
    if not HAS_OPENAI:
        click.echo("Error: openai package required. Install with: pip install openai")
        return

    from eval.runner import EvaluationRunner, EvaluationConfig

    # Get OpenAI API key
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        click.echo("Error: OPENAI_API_KEY environment variable required")
        return

    llm_client = OpenAI(api_key=openai_key)

    # Determine API mode
    use_agentforce = agent_id is not None
    if use_agentforce and not api_key:
        click.echo("Error: --api-key (access token) required for Agentforce API")
        return

    config = EvaluationConfig(
        casey_api_url=api_url,
        casey_api_key=api_key,
        agentforce_agent_id=agent_id,
        use_agentforce=use_agentforce,
        num_random_personas=conversations,
        include_edge_cases=include_edge_cases,
        judge_model=judge_model,
        synthetic_client_model=client_model,
    )

    api_mode = "Agentforce" if use_agentforce else "Legacy HTTP"
    click.echo(f"Starting evaluation run with {conversations} random + edge case personas...")
    click.echo(f"API Mode: {api_mode}")
    click.echo(f"API URL: {api_url}")
    if agent_id:
        click.echo(f"Agent ID: {agent_id}")

    def on_conversation(i: int, result):
        status = "✓" if result.completed else "✗"
        click.echo(f"  [{i+1}] {status} {result.persona.name} - {result.turn_count} turns")

    def on_evaluation(i: int, evaluation):
        if verbose:
            safety_pass = evaluation.passed_all_safety()
            quality = evaluation.get_quality_scores()
            click.echo(f"      Safety: {'PASS' if safety_pass else 'FAIL'}")
            if quality:
                click.echo(f"      Quality: {quality}")

    with EvaluationRunner(config, llm_client) as runner:
        summary = runner.run(
            on_conversation=on_conversation,
            on_evaluation=on_evaluation if verbose else None,
        )

    # Print summary
    click.echo("\n" + "=" * 60)
    click.echo("EVALUATION SUMMARY")
    click.echo("=" * 60)
    click.echo(f"Total conversations: {summary.total_conversations}")
    click.echo(f"Completed: {summary.completed_conversations}")
    click.echo(f"Failed: {summary.failed_conversations}")
    click.echo(f"\nBehavioral pass rate: {summary.behavioral_pass_rate:.1%}")
    click.echo(f"Safety pass rate: {summary.safety_pass_rate:.1%}")

    if summary.average_quality_scores:
        click.echo("\nQuality scores:")
        for judge, score in summary.average_quality_scores.items():
            click.echo(f"  {judge}: {score:.2f}/5")

    if summary.safety_failures:
        click.echo(f"\n⚠️  {len(summary.safety_failures)} safety failures:")
        for failure in summary.safety_failures[:5]:
            click.echo(f"  - {failure['persona']}: {failure['judge']}")

    # Save results
    if output:
        output_path = Path(output)
        output_path.write_text(summary.to_json())
        click.echo(f"\nResults saved to: {output}")
    else:
        default_output = f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path(default_output).write_text(summary.to_json())
        click.echo(f"\nResults saved to: {default_output}")


@cli.command()
@click.option("--persona", "-p", required=True, help="Persona name or ID")
@click.option("--api-url", envvar="CASEY_API_URL", required=True, help="Salesforce My Domain URL or legacy API URL")
@click.option("--api-key", envvar="CASEY_API_KEY", default=None, help="Access token (Agentforce) or API key (legacy)")
@click.option("--agent-id", envvar="AGENTFORCE_AGENT_ID", default=None, help="Agentforce 18-char agent ID")
@click.option("--preview", is_flag=True, help="Preview persona without running")
@click.option("--output", "-o", default=None, help="Output file for results")
def run_single(
    persona: str,
    api_url: str,
    api_key: str,
    agent_id: str,
    preview: bool,
    output: str,
):
    """Run evaluation for a single persona."""
    from eval.personas.edge_cases import get_persona_by_name, ALL_EDGE_CASE_PERSONAS

    # Find persona
    found_persona = get_persona_by_name(persona)

    if not found_persona:
        # Try matching by partial name
        for p in ALL_EDGE_CASE_PERSONAS:
            if persona.lower() in p.name.lower():
                found_persona = p
                break

    if not found_persona:
        click.echo(f"Error: Persona '{persona}' not found")
        click.echo("\nAvailable personas:")
        for p in ALL_EDGE_CASE_PERSONAS:
            click.echo(f"  - {p.name}")
        return

    click.echo(f"Persona: {found_persona.name}")
    click.echo(f"Language: {found_persona.primary_language.value}")
    click.echo(f"Legal Issue: {found_persona.legal_issue.value}")
    click.echo(f"Issue: {found_persona.issue_details}")

    if preview:
        click.echo("\n" + found_persona.get_system_prompt_context())
        return

    if not HAS_OPENAI:
        click.echo("Error: openai package required. Install with: pip install openai")
        return

    from eval.runner import EvaluationRunner, EvaluationConfig

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        click.echo("Error: OPENAI_API_KEY environment variable required")
        return

    llm_client = OpenAI(api_key=openai_key)

    # Determine API mode
    use_agentforce = agent_id is not None
    if use_agentforce and not api_key:
        click.echo("Error: --api-key (access token) required for Agentforce API")
        return

    config = EvaluationConfig(
        casey_api_url=api_url,
        casey_api_key=api_key,
        agentforce_agent_id=agent_id,
        use_agentforce=use_agentforce,
        num_random_personas=0,
        include_edge_cases=False,
    )

    api_mode = "Agentforce" if use_agentforce else "Legacy HTTP"
    click.echo(f"\nRunning conversation ({api_mode})...")

    with EvaluationRunner(config, llm_client) as runner:
        evaluation = runner.run_single(found_persona)

    # Print results
    click.echo("\n" + "=" * 60)
    click.echo("CONVERSATION")
    click.echo("=" * 60)
    click.echo(evaluation.conversation.get_transcript())

    click.echo("\n" + "=" * 60)
    click.echo("EVALUATION RESULTS")
    click.echo("=" * 60)
    click.echo(f"Completed: {evaluation.conversation.completed}")
    click.echo(f"Turns: {evaluation.conversation.turn_count}")
    click.echo(f"Duration: {evaluation.conversation.duration_seconds:.1f}s")

    if evaluation.safety_results:
        click.echo("\nSafety:")
        for r in evaluation.safety_results:
            icon = "✓" if r.verdict.value == "pass" else "✗" if r.verdict.value == "fail" else "-"
            click.echo(f"  {icon} {r.judge_id}: {r.verdict.value}")

    if evaluation.quality_results:
        click.echo("\nQuality:")
        for r in evaluation.quality_results:
            score = f"{r.score:.1f}/5" if r.score else r.verdict.value
            click.echo(f"  {r.judge_id}: {score}")

    if output:
        Path(output).write_text(json.dumps(evaluation.to_dict(), indent=2))
        click.echo(f"\nResults saved to: {output}")


@cli.command()
@click.option("--category", "-c", default=None, help="Filter by category (safety, behavioral, language)")
def list_personas(category: str):
    """List available edge case personas."""
    from eval.personas.edge_cases import (
        SAFETY_PERSONAS,
        BEHAVIORAL_PERSONAS,
        LANGUAGE_PERSONAS,
        ALL_EDGE_CASE_PERSONAS,
    )

    if category == "safety":
        personas = SAFETY_PERSONAS
    elif category == "behavioral":
        personas = BEHAVIORAL_PERSONAS
    elif category == "language":
        personas = LANGUAGE_PERSONAS
    else:
        personas = ALL_EDGE_CASE_PERSONAS

    click.echo(f"{'Name':<25} {'Language':<12} {'Issue':<15} {'Special'}")
    click.echo("-" * 70)

    for p in personas:
        specials = []
        if p.discloses_dv:
            specials.append("DV")
        if p.discloses_crisis:
            specials.append("Crisis")
        if p.gives_impossible_answers:
            specials.append("Impossible")
        if p.mentions_multiple_issues:
            specials.append("Multi-issue")
        if p.attempts_out_of_scope:
            specials.append("Out-of-scope")
        if p.is_returning_client:
            specials.append("Returning")

        click.echo(
            f"{p.name:<25} {p.primary_language.value:<12} {p.legal_issue.value:<15} {', '.join(specials)}"
        )


@cli.command()
@click.option("--category", "-c", default=None, help="Filter by category")
def list_rules(category: str):
    """List behavioral rules extracted from the prompt."""
    from eval.rules import BEHAVIORAL_RULES, RuleCategory, get_rules_by_category

    if category:
        try:
            cat = RuleCategory(category)
            rules = get_rules_by_category(cat)
        except ValueError:
            click.echo(f"Invalid category. Options: {[c.value for c in RuleCategory]}")
            return
    else:
        rules = BEHAVIORAL_RULES

    click.echo(f"{'ID':<30} {'Category':<15} {'Title'}")
    click.echo("-" * 80)

    for rule in rules:
        click.echo(f"{rule.id:<30} {rule.category.value:<15} {rule.title}")

    click.echo(f"\nTotal: {len(rules)} rules")


@cli.command()
def show_distributions():
    """Show target persona distributions."""
    from eval.personas.distributions import (
        LANGUAGE_DISTRIBUTION,
        LEGAL_ISSUE_DISTRIBUTION,
        validate_all_distributions,
    )

    click.echo("LANGUAGE DISTRIBUTION")
    click.echo("-" * 40)
    for lang, pct in LANGUAGE_DISTRIBUTION.items():
        bar = "█" * int(pct * 40)
        click.echo(f"{lang.value:<12} {bar} {pct:.0%}")

    click.echo("\nLEGAL ISSUE DISTRIBUTION")
    click.echo("-" * 40)
    for issue, pct in LEGAL_ISSUE_DISTRIBUTION.items():
        bar = "█" * int(pct * 40)
        click.echo(f"{issue.value:<20} {bar} {pct:.0%}")

    # Validate
    validation = validate_all_distributions()
    invalid = [k for k, v in validation.items() if not v]
    if invalid:
        click.echo(f"\n⚠️  Invalid distributions (don't sum to 100%): {invalid}")


@cli.command()
@click.argument("results_file")
def report(results_file: str):
    """Generate report from evaluation results."""
    results_path = Path(results_file)
    if not results_path.exists():
        click.echo(f"Error: File not found: {results_file}")
        return

    data = json.loads(results_path.read_text())

    click.echo("=" * 60)
    click.echo("CASEY EVALUATION REPORT")
    click.echo(f"Run ID: {data.get('run_id', 'unknown')}")
    click.echo(f"Timestamp: {data.get('timestamp', 'unknown')}")
    click.echo("=" * 60)

    click.echo(f"\nCONVERSATIONS")
    click.echo("-" * 40)
    click.echo(f"Total:     {data['total_conversations']}")
    click.echo(f"Completed: {data['completed_conversations']}")
    click.echo(f"Failed:    {data['failed_conversations']}")

    click.echo(f"\nBEHAVIORAL COMPLIANCE")
    click.echo("-" * 40)
    click.echo(f"Overall pass rate: {data['behavioral_pass_rate']:.1%}")

    if data.get('behavioral_by_category'):
        for cat, stats in data['behavioral_by_category'].items():
            rate = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
            click.echo(f"  {cat}: {rate:.1%} ({stats['passed']}/{stats['total']})")

    click.echo(f"\nSAFETY")
    click.echo("-" * 40)
    click.echo(f"Pass rate: {data['safety_pass_rate']:.1%}")

    if data.get('safety_failures'):
        click.echo(f"Failures ({len(data['safety_failures'])}):")
        for f in data['safety_failures'][:5]:
            click.echo(f"  ⚠️  {f['persona']}: {f['judge']}")

    click.echo(f"\nQUALITY SCORES")
    click.echo("-" * 40)
    if data.get('average_quality_scores'):
        for judge, score in data['average_quality_scores'].items():
            bar = "█" * int(score)
            click.echo(f"{judge:<30} {bar} {score:.2f}/5")


if __name__ == "__main__":
    cli()
