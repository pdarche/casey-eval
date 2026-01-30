#!/usr/bin/env python3
"""
Run a single simulated conversation using the Agentforce SDK.

Usage:
    uv run python run_conversation.py
    uv run python run_conversation.py --persona "Maria Santos"
    uv run python run_conversation.py --max-turns 20
    uv run python run_conversation.py --output transcripts/
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_persona(persona_name: str = None):
    """Get a persona by name or return a default."""
    from eval.personas.edge_cases import ALL_EDGE_CASE_PERSONAS, get_persona_by_name

    if persona_name:
        persona = get_persona_by_name(persona_name)
        if not persona:
            # Try partial match
            for p in ALL_EDGE_CASE_PERSONAS:
                if persona_name.lower() in p.name.lower():
                    return p
            print(f"Persona '{persona_name}' not found. Available:")
            for p in ALL_EDGE_CASE_PERSONAS:
                print(f"  - {p.name}")
            sys.exit(1)
        return persona

    # Default: Maria Santos (DV disclosure case)
    return get_persona_by_name("Maria Santos") or ALL_EDGE_CASE_PERSONAS[0]


def run_conversation(persona, max_turns: int = 50, verbose: bool = True, output_dir: str = None):
    """Run a simulated conversation between synthetic client and Casey."""
    from agentforce.agents import Agentforce
    from openai import OpenAI
    from eval.simulation.client import SyntheticClient

    # Get credentials
    salesforce_org = os.environ["CASEY_API_URL"].replace("https://", "")
    client_id = os.environ["SALESFORCE_CLIENT_ID"]
    client_secret = os.environ["SALESFORCE_CLIENT_SECRET"]
    agent_id = os.environ["AGENTFORCE_AGENT_ID"]
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not openai_key:
        print("Error: OPENAI_API_KEY required")
        sys.exit(1)

    # Initialize clients
    agent = Agentforce()
    openai_client = OpenAI(api_key=openai_key)
    synthetic_client = SyntheticClient(
        persona=persona,
        llm_client=openai_client,
        model="gpt-4.1-mini",
    )

    # Transcript storage
    transcript = []
    start_time = datetime.now()

    print("=" * 60)
    print(f"PERSONA: {persona.name}")
    print(f"Language: {persona.primary_language.value}")
    print(f"Issue: {persona.legal_issue.value}")
    print(f"Details: {persona.issue_details}")
    print("=" * 60)
    print()

    # Authenticate
    if verbose:
        print("Authenticating with Salesforce...")
    agent.authenticate(
        salesforce_org=salesforce_org,
        client_id=client_id,
        client_secret=client_secret,
    )

    # Start session
    if verbose:
        print("Starting session...")
    session = agent.start_session(agent_id=agent_id)
    session_id = session.sessionId

    # Get initial greeting
    agent_message = ""
    if session.messages:
        agent_message = session.messages[0].message
        print(f"\nCASEY: {agent_message}\n")
        transcript.append({"role": "casey", "content": agent_message})

    turn_count = 0
    completion_reason = "max_turns"

    try:
        while turn_count < max_turns:
            turn_count += 1

            # Generate client response
            client_response = synthetic_client.generate_response(agent_message)
            print(f"CLIENT: {client_response}\n")
            transcript.append({"role": "client", "content": client_response})

            # Check for completion signal
            if "INTAKE_COMPLETE" in client_response.upper():
                print("\n[Intake marked complete by synthetic client]")
                completion_reason = "intake_complete"
                break

            # Send to Casey
            agent.add_message_text(client_response)
            try:
                response = agent.send_message(session_id=session_id)
            except KeyError as e:
                # SDK doesn't handle some response formats (missing planId, etc.)
                print(f"\n[SDK parsing error: {e} - conversation may be complete]")
                completion_reason = "sdk_error"
                break

            if response.messages:
                agent_message = response.messages[0].message
                print(f"CASEY: {agent_message}\n")
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
                    # One more turn for client to acknowledge
                    client_response = synthetic_client.generate_response(agent_message)
                    print(f"CLIENT: {client_response}\n")
                    transcript.append({"role": "client", "content": client_response})
                    print("\n[Intake complete - Casey indicated follow-up]")
                    completion_reason = "intake_complete"
                    break
            else:
                print("[No response from Casey]")
                completion_reason = "no_response"
                break

    finally:
        # End session
        if verbose:
            print("\nEnding session...")
        try:
            agent.end_session(session_id=session_id)
        except Exception as e:
            print(f"Warning: Failed to end session: {e}")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("=" * 60)
    print(f"Conversation completed: {turn_count} turns ({duration:.1f}s)")
    print("=" * 60)

    # Build result object
    result = {
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
        "turn_count": turn_count,
        "completion_reason": completion_reason,
        "transcript": transcript,
    }

    # Save transcript if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        safe_name = persona.name.replace(" ", "_").lower()
        filename = f"{timestamp}_{safe_name}.json"
        filepath = output_path / filename

        filepath.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nTranscript saved to: {filepath}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run a simulated conversation with Casey")
    parser.add_argument("--persona", "-p", help="Persona name to use")
    parser.add_argument("--max-turns", "-m", type=int, default=50, help="Maximum turns")
    parser.add_argument("--output", "-o", default="transcripts", help="Output directory for transcripts")
    parser.add_argument("--no-save", action="store_true", help="Don't save transcript to file")
    parser.add_argument("--list", "-l", action="store_true", help="List available personas")
    parser.add_argument("--quiet", "-q", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    if args.list:
        from eval.personas.edge_cases import ALL_EDGE_CASE_PERSONAS
        print("Available personas:")
        for p in ALL_EDGE_CASE_PERSONAS:
            flags = []
            if p.discloses_dv:
                flags.append("DV")
            if p.discloses_crisis:
                flags.append("Crisis")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"  - {p.name} ({p.primary_language.value}){flag_str}")
        return

    persona = get_persona(args.persona)
    output_dir = None if args.no_save else args.output
    run_conversation(persona, max_turns=args.max_turns, verbose=not args.quiet, output_dir=output_dir)


if __name__ == "__main__":
    main()
