#!/usr/bin/env python3
"""
Run multiple simulated conversations in parallel.

Usage:
    uv run python scripts/run_batch.py --count 10
    uv run python scripts/run_batch.py --count 10 --parallel 5
    uv run python scripts/run_batch.py --version "v1.0-baseline" --count 10
"""

import os
import sys
import json
import argparse
import concurrent.futures
from datetime import datetime
from pathlib import Path
from threading import Lock
from dotenv import load_dotenv

load_dotenv()

# Lock for thread-safe printing
print_lock = Lock()


def safe_print(*args, **kwargs):
    """Thread-safe print."""
    with print_lock:
        print(*args, **kwargs)


def run_single_conversation(persona, max_turns: int, output_dir: str, conversation_id: int, version: str = None):
    """Run a single conversation (designed to be called in parallel)."""
    from agentforce.agents import Agentforce
    from openai import OpenAI
    from eval.simulation.client import SyntheticClient

    # Get credentials
    salesforce_org = os.environ["CASEY_API_URL"].replace("https://", "")
    client_id = os.environ["SALESFORCE_CLIENT_ID"]
    client_secret = os.environ["SALESFORCE_CLIENT_SECRET"]
    agent_id = os.environ["AGENTFORCE_AGENT_ID"]
    openai_key = os.environ.get("OPENAI_API_KEY")

    # Initialize clients (each thread gets its own instances)
    agent = Agentforce()
    openai_client = OpenAI(api_key=openai_key)
    synthetic_client = SyntheticClient(
        persona=persona,
        llm_client=openai_client,
        model="gpt-4.1-mini",
    )

    transcript = []
    start_time = datetime.now()
    session_id = ""
    completion_reason = "max_turns"
    error = None

    safe_print(f"[{conversation_id}] Starting: {persona.name} ({persona.legal_issue.value})")

    try:
        # Authenticate
        agent.authenticate(
            salesforce_org=salesforce_org,
            client_id=client_id,
            client_secret=client_secret,
        )

        # Start session
        session = agent.start_session(agent_id=agent_id)
        session_id = session.sessionId

        # Get initial greeting
        agent_message = ""
        if session.messages:
            agent_message = session.messages[0].message
            transcript.append({"role": "casey", "content": agent_message})

        turn_count = 0

        while turn_count < max_turns:
            turn_count += 1

            # Generate client response
            client_response = synthetic_client.generate_response(agent_message)
            transcript.append({"role": "client", "content": client_response})

            # Check for completion signal
            if "INTAKE_COMPLETE" in client_response.upper():
                completion_reason = "intake_complete"
                break

            # Send to Casey
            agent.add_message_text(client_response)
            try:
                response = agent.send_message(session_id=session_id)
            except KeyError as e:
                completion_reason = "sdk_error"
                break

            if response.messages:
                agent_message = response.messages[0].message
                transcript.append({"role": "casey", "content": agent_message})

                # Check for completion indicators
                completion_phrases = [
                    "intake is complete",
                    "form is complete",
                    "thank you for completing",
                    "we'll follow up",
                    "within 2 business days",
                ]
                if any(phrase in agent_message.lower() for phrase in completion_phrases):
                    client_response = synthetic_client.generate_response(agent_message)
                    transcript.append({"role": "client", "content": client_response})
                    completion_reason = "intake_complete"
                    break
            else:
                completion_reason = "no_response"
                break

        # End session
        try:
            agent.end_session(session_id=session_id)
        except Exception:
            pass

    except Exception as e:
        completion_reason = "error"
        error = str(e)
        safe_print(f"[{conversation_id}] Error: {e}")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Build result
    result = {
        "version": version,
        "persona": {
            "name": persona.name,
            "language": persona.primary_language.value,
            "legal_issue": persona.legal_issue.value,
            "issue_details": persona.issue_details,
            "discloses_dv": persona.discloses_dv,
            "discloses_crisis": persona.discloses_crisis,
        },
        "session_id": session_id,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "turn_count": len([t for t in transcript if t["role"] == "client"]),
        "completion_reason": completion_reason,
        "error": error,
        "transcript": transcript,
    }

    # Save transcript
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        safe_name = persona.name.replace(" ", "_").lower()
        filename = f"{timestamp}_{safe_name}_{conversation_id}.json"
        filepath = output_path / filename

        filepath.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    status = "✓" if completion_reason == "intake_complete" else "✗" if error else "~"
    safe_print(f"[{conversation_id}] {status} {persona.name}: {result['turn_count']} turns, {duration:.1f}s ({completion_reason})")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run multiple conversations in parallel")
    parser.add_argument("--count", "-n", type=int, default=10, help="Number of conversations to run")
    parser.add_argument("--parallel", "-p", type=int, default=10, help="Max parallel conversations")
    parser.add_argument("--max-turns", "-m", type=int, default=50, help="Max turns per conversation")
    parser.add_argument("--output", "-o", default="transcripts", help="Output directory")
    parser.add_argument("--all-personas", action="store_true", help="Run all edge case personas")
    parser.add_argument("--random", action="store_true", help="Generate random personas")
    parser.add_argument("--version", "-v", default=None, help="Version tag for this eval run (e.g., 'v1.0-baseline')")

    args = parser.parse_args()

    # Generate version from timestamp if not provided
    version = args.version or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check for required env vars
    required = ["CASEY_API_URL", "SALESFORCE_CLIENT_ID", "SALESFORCE_CLIENT_SECRET",
                "AGENTFORCE_AGENT_ID", "OPENAI_API_KEY"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"Error: Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    # Get personas
    from eval.personas.edge_cases import ALL_EDGE_CASE_PERSONAS
    from eval.personas.generator import PersonaGenerator

    if args.all_personas:
        personas = ALL_EDGE_CASE_PERSONAS
    elif args.random:
        generator = PersonaGenerator()
        personas = generator.generate_batch(args.count)
    else:
        # Cycle through edge case personas
        personas = []
        for i in range(args.count):
            personas.append(ALL_EDGE_CASE_PERSONAS[i % len(ALL_EDGE_CASE_PERSONAS)])

    print("=" * 60)
    print(f"BATCH EVALUATION")
    print(f"Version: {version}")
    print(f"Conversations: {len(personas)}")
    print(f"Parallel workers: {args.parallel}")
    print(f"Max turns: {args.max_turns}")
    print(f"Output: {args.output}")
    print("=" * 60)
    print()

    start_time = datetime.now()
    results = []

    # Run conversations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for i, persona in enumerate(personas):
            future = executor.submit(
                run_single_conversation,
                persona,
                args.max_turns,
                args.output,
                i + 1,
                version,
            )
            futures.append(future)

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error in conversation: {e}")

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    # Print summary
    print()
    print("=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"Total conversations: {len(results)}")
    print(f"Completed (intake_complete): {sum(1 for r in results if r['completion_reason'] == 'intake_complete')}")
    print(f"Max turns reached: {sum(1 for r in results if r['completion_reason'] == 'max_turns')}")
    print(f"Errors: {sum(1 for r in results if r['completion_reason'] == 'error')}")
    print(f"Total time: {total_duration:.1f}s")
    print(f"Avg time per conversation: {total_duration / len(results):.1f}s")

    # Save batch summary
    summary_path = Path(args.output) / f"batch_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "version": version,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_duration_seconds": total_duration,
        "conversation_count": len(results),
        "completion_reasons": {
            "intake_complete": sum(1 for r in results if r["completion_reason"] == "intake_complete"),
            "max_turns": sum(1 for r in results if r["completion_reason"] == "max_turns"),
            "error": sum(1 for r in results if r["completion_reason"] == "error"),
            "sdk_error": sum(1 for r in results if r["completion_reason"] == "sdk_error"),
            "no_response": sum(1 for r in results if r["completion_reason"] == "no_response"),
        },
        "conversations": [
            {
                "persona": r["persona"]["name"],
                "turn_count": r["turn_count"],
                "duration_seconds": r["duration_seconds"],
                "completion_reason": r["completion_reason"],
            }
            for r in results
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
