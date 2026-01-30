#!/usr/bin/env python3
"""
Run judges on a saved transcript.

Usage:
    uv run python run_judges.py transcripts/20260129_163339_patricia_johnson.json
    uv run python run_judges.py transcripts/*.json  # evaluate all
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def load_transcript(filepath: str) -> dict:
    """Load a transcript JSON file."""
    return json.loads(Path(filepath).read_text())


def transcript_to_context(transcript_data: dict):
    """Convert transcript JSON to ConversationContext."""
    from eval.judges.base import ConversationContext, ConversationTurn

    turns = []
    for msg in transcript_data.get("transcript", []):
        role = "assistant" if msg["role"] == "casey" else "user"
        turns.append(ConversationTurn(role=role, content=msg["content"]))

    persona = transcript_data.get("persona", {})

    return ConversationContext(
        turns=turns,
        persona=persona,
    )


def run_judges(context, llm_client, judge_model: str = "gpt-4o"):
    """Run all judges on the conversation context."""
    from eval.judges.safety import SafetyEvaluator
    from eval.judges.quality import QualityEvaluator

    results = {
        "safety": [],
        "quality": [],
    }

    # Run safety judges
    print("\nRunning Safety Judges...")
    safety_eval = SafetyEvaluator(llm_client=llm_client, model=judge_model)
    safety_results = safety_eval.evaluate_all(context)
    results["safety"] = safety_results

    for r in safety_results:
        icon = "✓" if r.verdict.value == "pass" else "✗" if r.verdict.value == "fail" else "-"
        print(f"  {icon} {r.judge_id}: {r.verdict.value}")
        if r.reasoning:
            print(f"      {r.reasoning[:100]}...")

    # Run quality judges
    print("\nRunning Quality Judges...")
    quality_eval = QualityEvaluator(llm_client=llm_client, model=judge_model)
    quality_results = quality_eval.evaluate_all(context)
    results["quality"] = quality_results

    for r in quality_results:
        score_str = f"{r.score:.1f}/5" if r.score else r.verdict.value
        icon = "✓" if r.verdict.value == "pass" else "✗" if r.verdict.value == "fail" else "~"
        print(f"  {icon} {r.judge_id}: {score_str}")
        if r.reasoning:
            print(f"      {r.reasoning[:100]}...")

    return results


def print_summary(results: dict):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    # Safety summary
    safety_results = results["safety"]
    safety_passed = sum(1 for r in safety_results if r.verdict.value == "pass")
    safety_failed = sum(1 for r in safety_results if r.verdict.value == "fail")
    safety_na = sum(1 for r in safety_results if r.verdict.value == "not_applicable")

    print(f"\nSafety: {safety_passed} passed, {safety_failed} failed, {safety_na} n/a")

    if safety_failed > 0:
        print("  ⚠️  Safety failures:")
        for r in safety_results:
            if r.verdict.value == "fail":
                print(f"    - {r.judge_id}: {r.reasoning}")

    # Quality summary
    quality_results = results["quality"]
    scores = [r.score for r in quality_results if r.score is not None]
    avg_score = sum(scores) / len(scores) if scores else 0

    print(f"\nQuality: avg {avg_score:.1f}/5")
    for r in quality_results:
        if r.score is not None:
            print(f"  - {r.judge_id}: {r.score:.1f}/5")


def main():
    parser = argparse.ArgumentParser(description="Run judges on saved transcripts")
    parser.add_argument("transcripts", nargs="+", help="Transcript JSON file(s)")
    parser.add_argument("--model", "-m", default="gpt-4o", help="Judge model")
    parser.add_argument("--output", "-o", help="Output file for results JSON")

    args = parser.parse_args()

    # Check for OpenAI key
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("Error: OPENAI_API_KEY required")
        sys.exit(1)

    from openai import OpenAI
    llm_client = OpenAI(api_key=openai_key)

    all_results = []

    for filepath in args.transcripts:
        print("=" * 60)
        print(f"Evaluating: {filepath}")
        print("=" * 60)

        # Load transcript
        transcript_data = load_transcript(filepath)
        persona = transcript_data.get("persona", {})
        print(f"Persona: {persona.get('name', 'Unknown')}")
        print(f"Turns: {transcript_data.get('turn_count', 0)}")
        print(f"Completion: {transcript_data.get('completion_reason', 'unknown')}")

        # Convert to context
        context = transcript_to_context(transcript_data)

        # Run judges
        results = run_judges(context, llm_client, args.model)

        # Print summary
        print_summary(results)

        # Collect for output
        all_results.append({
            "file": filepath,
            "persona": persona,
            "safety": [r.to_dict() for r in results["safety"]],
            "quality": [r.to_dict() for r in results["quality"]],
        })

    # Save results if requested
    if args.output:
        Path(args.output).write_text(json.dumps(all_results, indent=2))
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
