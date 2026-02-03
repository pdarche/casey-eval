#!/usr/bin/env python3
"""
Flask webapp for viewing Casey evaluation results.

Usage:
    uv run python webapp/app.py
"""

import json
import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_login import login_required

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from webapp.utils import (
    list_versions,
    get_version,
    get_conversation,
    get_conversation_by_file,
    get_transcripts_dir,
)
from webapp.auth import init_auth

app = Flask(__name__)
init_auth(app)


# =============================================================================
# Prompt Management Routes
# =============================================================================

def get_db_available():
    """Check if database is available."""
    try:
        from eval.database import init_pool, get_cursor
        init_pool()
        with get_cursor() as cursor:
            cursor.execute("SELECT 1")
        return True
    except Exception:
        return False


@app.route("/prompts")
@login_required
def prompts_list():
    """List all prompt versions."""
    if not get_db_available():
        return render_template("prompts.html", prompts=[], error="Database not available")

    from eval.database import list_prompt_versions
    prompts = list_prompt_versions(active_only=False)
    return render_template("prompts.html", prompts=prompts)


@app.route("/prompts/<int:prompt_id>")
@login_required
def prompt_detail(prompt_id: int):
    """View a specific prompt version."""
    if not get_db_available():
        return "Database not available", 500

    from eval.database import get_prompt_version_by_id
    prompt = get_prompt_version_by_id(prompt_id)
    if not prompt:
        return "Prompt not found", 404

    return render_template("prompt_detail.html", prompt=prompt, metadata=prompt.metadata)


@app.route("/prompts/<int:prompt_id>/clone")
@login_required
def prompt_clone(prompt_id: int):
    """Clone a prompt version."""
    if not get_db_available():
        return "Database not available", 500

    from eval.database import get_prompt_version_by_id
    source = get_prompt_version_by_id(prompt_id)
    if not source:
        return "Prompt not found", 404

    return render_template("prompt_clone.html", source=source)


@app.route("/api/prompts", methods=["POST"])
@login_required
def create_prompt():
    """Create a new prompt version (API endpoint)."""
    if not get_db_available():
        return jsonify({"error": "Database not available"}), 500

    from eval.database import create_prompt_version, set_active_prompt_version, get_prompt_version

    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    version = data.get("version")
    content = data.get("content")

    if not version:
        return jsonify({"error": "Version is required"}), 400
    if not content:
        return jsonify({"error": "Content is required"}), 400

    # Check if version already exists
    existing = get_prompt_version(version)
    if existing:
        return jsonify({"error": f"Version '{version}' already exists"}), 400

    try:
        # Start with user-provided metadata
        metadata = data.get("metadata", {})

        # Extract intake steps from prompt content
        try:
            from eval.prompt_parser import extract_intake_steps
            steps_data = extract_intake_steps(content)
            metadata["intake_steps"] = steps_data.get("intake_steps", {})
            metadata["intake_steps_count"] = steps_data.get("steps_count", 0)
            metadata["intake_steps_extraction_model"] = steps_data.get("extraction_model", "")
        except Exception as e:
            # Log but don't fail - steps extraction is optional
            print(f"Warning: Failed to extract intake steps: {e}")

        prompt_id = create_prompt_version(
            version=version,
            name=data.get("name"),
            content=content,
            metadata=metadata,
            is_active=False,  # Create as inactive first
        )

        # If is_active is requested, set it active (this deactivates others)
        if data.get("is_active"):
            set_active_prompt_version(version)

        return jsonify({"id": prompt_id, "version": version})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/prompts/<int:prompt_id>/activate", methods=["POST"])
@login_required
def activate_prompt(prompt_id: int):
    """Set a prompt version as active (API endpoint)."""
    if not get_db_available():
        return jsonify({"error": "Database not available"}), 500

    from eval.database import get_prompt_version_by_id, set_active_prompt_version

    prompt = get_prompt_version_by_id(prompt_id)
    if not prompt:
        return jsonify({"error": "Prompt not found"}), 404

    try:
        set_active_prompt_version(prompt.version)
        return jsonify({"success": True, "version": prompt.version})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/prompts/<int:prompt_id>", methods=["DELETE"])
@login_required
def delete_prompt(prompt_id: int):
    """Delete a prompt version (API endpoint)."""
    if not get_db_available():
        return jsonify({"error": "Database not available"}), 500

    from eval.database import get_prompt_version_by_id, delete_prompt_version

    prompt = get_prompt_version_by_id(prompt_id)
    if not prompt:
        return jsonify({"error": "Prompt not found"}), 404

    # Don't allow deleting the active prompt
    if prompt.is_active:
        return jsonify({"error": "Cannot delete the active prompt version. Set another version as active first."}), 400

    try:
        delete_prompt_version(prompt_id)
        return jsonify({"success": True, "version": prompt.version})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# Evaluation Runs Management
# =============================================================================

@app.route("/runs")
@login_required
def runs_list():
    """List all evaluation runs."""
    if not get_db_available():
        return render_template("runs.html", runs=[], prompts=[], error="Database not available")

    from eval.database import list_simulation_runs, list_prompt_versions, get_cursor

    runs = list_simulation_runs(limit=100)
    prompts = list_prompt_versions(active_only=False)

    # Get conversation counts and metrics for each run
    runs_with_counts = []
    with get_cursor() as cursor:
        for run in runs:
            cursor.execute(
                "SELECT COUNT(*) as count FROM conversations WHERE simulation_run_id = %s",
                (run.id,)
            )
            count = cursor.fetchone()["count"]
            target_count = run.config.get("count", 10) if run.config else 10

            # Get metrics for completed runs
            safety_pass_rate = None
            avg_quality_score = None
            avg_completeness = None

            if run.status == 'completed' and count > 0:
                # Safety pass rate
                cursor.execute(
                    """SELECT
                        COUNT(CASE WHEN j.verdict = 'pass' THEN 1 END)::float /
                        NULLIF(COUNT(*), 0) as pass_rate
                    FROM judgments j
                    JOIN conversations c ON j.conversation_id = c.id
                    WHERE c.simulation_run_id = %s AND j.judge_type = 'safety'""",
                    (run.id,)
                )
                result = cursor.fetchone()
                safety_pass_rate = result["pass_rate"] if result else None

                # Average quality score
                cursor.execute(
                    """SELECT AVG(j.score) as avg_score
                    FROM judgments j
                    JOIN conversations c ON j.conversation_id = c.id
                    WHERE c.simulation_run_id = %s AND j.judge_type = 'quality'""",
                    (run.id,)
                )
                result = cursor.fetchone()
                avg_quality_score = result["avg_score"] if result else None

                # Average completeness
                cursor.execute(
                    """SELECT AVG(j.score) as avg_score
                    FROM judgments j
                    JOIN conversations c ON j.conversation_id = c.id
                    WHERE c.simulation_run_id = %s AND j.judge_type = 'completeness'""",
                    (run.id,)
                )
                result = cursor.fetchone()
                avg_completeness = result["avg_score"] if result else None

            # Check if run has judgments
            cursor.execute(
                """SELECT COUNT(*) as count FROM judgments j
                   JOIN conversations c ON j.conversation_id = c.id
                   WHERE c.simulation_run_id = %s""",
                (run.id,)
            )
            has_judgments = cursor.fetchone()["count"] > 0

            run_dict = {
                "id": run.id,
                "version": run.version,
                "status": run.status,
                "started_at": run.started_at,
                "completed_at": run.completed_at,
                "conversation_count": count,
                "target_count": target_count,
                "safety_pass_rate": safety_pass_rate,
                "avg_quality_score": avg_quality_score,
                "avg_completeness": avg_completeness,
                "has_judgments": has_judgments,
            }
            runs_with_counts.append(type('Run', (), run_dict)())

    return render_template("runs.html", runs=runs_with_counts, prompts=prompts)


@app.route("/api/runs", methods=["POST"])
@login_required
def create_run():
    """Create and start a new evaluation run (API endpoint)."""
    if not get_db_available():
        return jsonify({"error": "Database not available"}), 500

    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    version = data.get("version")
    if not version:
        return jsonify({"error": "Version is required"}), 400

    # Check required environment variables
    required_env = ["CASEY_API_URL", "SALESFORCE_CLIENT_ID", "SALESFORCE_CLIENT_SECRET",
                    "AGENTFORCE_AGENT_ID", "OPENAI_API_KEY"]
    missing = [v for v in required_env if not os.environ.get(v)]
    if missing:
        return jsonify({"error": f"Missing environment variables: {', '.join(missing)}"}), 500

    from eval.database import (
        create_simulation_run, update_simulation_run_status,
        get_simulation_run_by_version
    )

    # Check if version already exists
    existing = get_simulation_run_by_version(version)
    if existing:
        return jsonify({"error": f"Version '{version}' already exists"}), 400

    try:
        # Create the simulation run
        config = {
            "persona_set": data.get("persona_set", "all"),
            "count": data.get("count", 10),
            "max_turns": data.get("max_turns", 50),
            "parallel": data.get("parallel", 5),
            "client_model": data.get("client_model", "gpt-4.1-mini"),
            "judge_model": data.get("judge_model", "gpt-4.1"),
            "auto_judge": data.get("auto_judge", True),
        }

        run_id = create_simulation_run(
            version=version,
            config=config,
            prompt_version_id=data.get("prompt_version_id"),
            status="running",
        )

        # Start the evaluation in a background thread
        import threading
        thread = threading.Thread(
            target=run_evaluation_background,
            args=(run_id, version, config),
            daemon=True
        )
        thread.start()

        return jsonify({"id": run_id, "version": version, "status": "running"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def run_evaluation_background(run_id: int, version: str, config: dict):
    """Run evaluation in background thread."""
    from eval.database import (
        update_simulation_run_status, create_conversation,
        create_judgment, init_pool
    )
    from eval.personas.edge_cases import get_personas_by_tag, ALL_EDGE_CASE_PERSONAS
    from eval.personas.generator import PersonaGenerator
    from webapp.run_tracker import register_run, cleanup_run

    # Initialize database pool for this thread
    init_pool()

    try:
        # Get personas
        persona_set = config.get("persona_set", "all")
        count = config.get("count", 10)

        if persona_set == "random":
            generator = PersonaGenerator()
            personas = generator.generate_batch(count)
        else:
            base_personas = get_personas_by_tag(persona_set)
            if not base_personas:
                base_personas = ALL_EDGE_CASE_PERSONAS
            # Cycle through personas to reach count
            personas = []
            for i in range(count):
                personas.append(base_personas[i % len(base_personas)])

        # Register run for in-memory tracking
        register_run(run_id, personas)

        # Run conversations
        import concurrent.futures
        max_turns = config.get("max_turns", 50)
        parallel = config.get("parallel", 5)
        client_model = config.get("client_model", "gpt-4.1-mini")

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = []
            for i, persona in enumerate(personas):
                future = executor.submit(
                    run_single_conversation_to_db,
                    persona, max_turns, run_id, i + 1, version, client_model
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in conversation: {e}")

        # Update run status
        summary = {
            "total": len(results),
            "completed": sum(1 for r in results if r.get("completion_reason") == "intake_complete"),
            "errors": sum(1 for r in results if r.get("completion_reason") == "error"),
        }

        # Auto-run judges if configured
        if config.get("auto_judge", True):
            update_simulation_run_status(run_id, "judging", summary)
            judge_model = config.get("judge_model", "gpt-4o")
            run_judges_on_simulation(run_id, judge_model)

        update_simulation_run_status(run_id, "completed", summary)

        # Clean up in-memory tracking
        cleanup_run(run_id)

    except Exception as e:
        print(f"Evaluation error: {e}")
        update_simulation_run_status(run_id, "failed", {"error": str(e)})
        # Clean up in-memory tracking on failure too
        cleanup_run(run_id)


def run_single_conversation_to_db(persona, max_turns: int, simulation_run_id: int,
                                   conversation_id: int, version: str,
                                   client_model: str = "gpt-4.1-mini") -> dict:
    """Run a single conversation and save to database."""
    from agentforce.agents import Agentforce
    from openai import OpenAI
    from eval.simulation.client import SyntheticClient
    from eval.database import create_conversation
    from datetime import datetime
    from webapp.run_tracker import start_conversation, update_turn_count, complete_conversation

    # Mark conversation as running in tracker
    start_conversation(simulation_run_id, conversation_id, persona)

    # Get credentials
    salesforce_org = os.environ["CASEY_API_URL"].replace("https://", "")
    client_id = os.environ["SALESFORCE_CLIENT_ID"]
    client_secret = os.environ["SALESFORCE_CLIENT_SECRET"]
    agent_id = os.environ["AGENTFORCE_AGENT_ID"]
    openai_key = os.environ.get("OPENAI_API_KEY")

    # Initialize clients
    agent = Agentforce()
    openai_client = OpenAI(api_key=openai_key)
    synthetic_client = SyntheticClient(
        persona=persona,
        llm_client=openai_client,
        model=client_model,
    )

    transcript = []
    start_time = datetime.now()
    session_id = ""
    completion_reason = "max_turns"
    error = None

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

            # Update turn count in tracker
            update_turn_count(simulation_run_id, conversation_id, turn_count)

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
            except KeyError:
                completion_reason = "sdk_error"
                break

            if response.messages:
                agent_message = response.messages[0].message
                transcript.append({"role": "casey", "content": agent_message})

                # Check for completion indicators
                completion_phrases = [
                    "intake is complete", "form is complete", "thank you for completing",
                    "we'll follow up", "within 2 business days",
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

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Build persona dict
    persona_dict = {
        "name": persona.name,
        "language": persona.primary_language.value,
        "legal_issue": persona.legal_issue.value,
        "issue_details": persona.issue_details,
        "discloses_dv": getattr(persona, 'discloses_dv', False),
        "discloses_crisis": getattr(persona, 'discloses_crisis', False),
    }

    # Save to database
    create_conversation(
        session_id=session_id or f"conv_{simulation_run_id}_{conversation_id}",
        persona=persona_dict,
        transcript=transcript,
        simulation_run_id=simulation_run_id,
        completion_reason=completion_reason,
        turn_count=len([t for t in transcript if t["role"] == "client"]),
        duration_seconds=duration,
        start_time=start_time,
        end_time=end_time,
        metadata={"error": error} if error else {},
    )

    # Mark conversation as complete in tracker
    complete_conversation(simulation_run_id, conversation_id)

    return {
        "completion_reason": completion_reason,
        "turn_count": len([t for t in transcript if t["role"] == "client"]),
        "duration_seconds": duration,
    }


def run_judges_on_simulation(simulation_run_id: int, judge_model: str = "gpt-4.1"):
    """Run judges on all conversations in a simulation."""
    from eval.database import (
        list_conversations_by_simulation, create_judgment, get_cursor, get_simulation_run
    )
    from eval.judges.base import ConversationContext, ConversationTurn
    from eval.judges.safety import SafetyEvaluator
    from eval.judges.quality import QualityEvaluator
    from eval.judges.completeness import CompletenessEvaluator
    from openai import OpenAI

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        return

    llm_client = OpenAI(api_key=openai_key)

    # Get the simulation run to find the prompt version
    sim_run = get_simulation_run(simulation_run_id)
    prompt_version_id = sim_run.prompt_version_id if sim_run else None

    conversations = list_conversations_by_simulation(simulation_run_id)

    # Clear existing judgments for this simulation (for re-judging)
    with get_cursor() as cursor:
        cursor.execute(
            """DELETE FROM judgments
               WHERE conversation_id IN (
                   SELECT id FROM conversations WHERE simulation_run_id = %s
               )""",
            (simulation_run_id,)
        )

    for conv in conversations:
        # Convert to context
        turns = []
        for msg in conv.transcript:
            role = "assistant" if msg["role"] == "casey" else "user"
            turns.append(ConversationTurn(role=role, content=msg["content"]))

        context = ConversationContext(turns=turns, persona=conv.persona)

        # Run judges
        safety_eval = SafetyEvaluator(llm_client=llm_client, model=judge_model)
        quality_eval = QualityEvaluator(llm_client=llm_client, model=judge_model)
        completeness_eval = CompletenessEvaluator(llm_client=llm_client, model=judge_model, prompt_version_id=prompt_version_id)

        # Safety judgments
        for result in safety_eval.evaluate_all(context):
            create_judgment(
                conversation_id=conv.id,
                judge_type="safety",
                judge_id=result.judge_id,
                verdict=result.verdict.value,
                score=result.score,
                reasoning=result.reasoning,
                evidence=result.evidence,
                metadata=result.metadata or {},
            )

        # Quality judgments
        for result in quality_eval.evaluate_all(context):
            create_judgment(
                conversation_id=conv.id,
                judge_type="quality",
                judge_id=result.judge_id,
                verdict=result.verdict.value,
                score=result.score,
                reasoning=result.reasoning,
                evidence=result.evidence,
                metadata=result.metadata or {},
            )

        # Completeness judgment
        result = completeness_eval.evaluate(context)
        create_judgment(
            conversation_id=conv.id,
            judge_type="completeness",
            judge_id=result.judge_id,
            verdict=result.verdict.value,
            score=result.score,
            reasoning=result.reasoning,
            evidence=result.evidence,
            metadata=result.metadata or {},
        )


def get_conversations_for_run_status(run_id: int, run_status: str = "running") -> list:
    """Merge in-memory active + DB completed conversations."""
    from webapp.run_tracker import get_run_progress
    from eval.database import get_cursor

    progress = get_run_progress(run_id)
    if not progress:
        return []

    # Get completed conversations from DB
    completed = {}
    judged_conversation_ids = set()

    with get_cursor() as cursor:
        cursor.execute(
            """SELECT id, persona, completion_reason, turn_count
               FROM conversations
               WHERE simulation_run_id = %s
               ORDER BY id""",
            (run_id,)
        )
        for i, row in enumerate(cursor.fetchall()):
            idx = i + 1
            completed[idx] = {
                "conversation_id": row["id"],
                "completion_reason": row["completion_reason"],
                "turn_count": row["turn_count"],
                "persona_name": row["persona"].get("name", f"Persona {idx}") if row["persona"] else f"Persona {idx}",
            }

        # If judging, check which conversations have judgments
        if run_status == "judging":
            cursor.execute(
                """SELECT DISTINCT conversation_id
                   FROM judgments j
                   JOIN conversations c ON j.conversation_id = c.id
                   WHERE c.simulation_run_id = %s""",
                (run_id,)
            )
            judged_conversation_ids = {row["conversation_id"] for row in cursor.fetchall()}

    # Build unified list
    conversations = []
    for i, persona in enumerate(progress["personas"]):
        idx = i + 1
        if idx in progress["completed_indices"]:
            # Get from DB
            conv = completed.get(idx, {})
            conv_id = conv.get("conversation_id")
            is_judged = conv_id in judged_conversation_ids if conv_id else False

            conversations.append({
                "index": idx,
                "persona_name": conv.get("persona_name", persona.get("name", f"Persona {idx}")),
                "status": "completed" if conv.get("completion_reason") != "error" else "error",
                "completion_reason": conv.get("completion_reason"),
                "turn_count": conv.get("turn_count", 0),
                "judged": is_judged,
            })
        elif idx in progress["active"]:
            # Currently running
            active = progress["active"][idx]
            conversations.append({
                "index": idx,
                "persona_name": active.get("persona_name", f"Persona {idx}"),
                "status": "running",
                "turn_count": active.get("turn_count", 0),
                "judged": False,
            })
        else:
            # Queued
            conversations.append({
                "index": idx,
                "persona_name": persona.get("name", f"Persona {idx}"),
                "status": "queued",
                "turn_count": 0,
                "judged": False,
            })

    return conversations


@app.route("/api/runs/status")
@login_required
def get_runs_status():
    """Get status of all runs for polling updates."""
    if not get_db_available():
        return jsonify([])

    from eval.database import list_simulation_runs, get_cursor

    # Check if conversation details are requested
    include_convs = request.args.get("include_conversations") == "true"
    expanded_run_ids = request.args.get("run_ids", "").split(",") if include_convs else []
    expanded_run_ids = [r for r in expanded_run_ids if r]  # Filter empty strings

    runs = list_simulation_runs(limit=50)
    result = []

    with get_cursor() as cursor:
        for run in runs:
            cursor.execute(
                "SELECT COUNT(*) as count FROM conversations WHERE simulation_run_id = %s",
                (run.id,)
            )
            conv_count = cursor.fetchone()["count"]

            # Get count of conversations that have been judged
            cursor.execute(
                """SELECT COUNT(DISTINCT j.conversation_id) as count
                   FROM judgments j
                   JOIN conversations c ON j.conversation_id = c.id
                   WHERE c.simulation_run_id = %s""",
                (run.id,)
            )
            judged_count = cursor.fetchone()["count"]

            # Get target count from config
            target_count = run.config.get("count", 10) if run.config else 10

            # Get metrics for completed runs
            safety_pass_rate = None
            avg_quality_score = None
            avg_completeness = None

            if run.status == 'completed' and conv_count > 0:
                # Safety pass rate
                cursor.execute(
                    """SELECT
                        COUNT(CASE WHEN j.verdict = 'pass' THEN 1 END)::float /
                        NULLIF(COUNT(*), 0) as pass_rate
                    FROM judgments j
                    JOIN conversations c ON j.conversation_id = c.id
                    WHERE c.simulation_run_id = %s AND j.judge_type = 'safety'""",
                    (run.id,)
                )
                row = cursor.fetchone()
                safety_pass_rate = row["pass_rate"] if row else None

                # Average quality score
                cursor.execute(
                    """SELECT AVG(j.score) as avg_score
                    FROM judgments j
                    JOIN conversations c ON j.conversation_id = c.id
                    WHERE c.simulation_run_id = %s AND j.judge_type = 'quality'""",
                    (run.id,)
                )
                row = cursor.fetchone()
                avg_quality_score = row["avg_score"] if row else None

                # Average completeness
                cursor.execute(
                    """SELECT AVG(j.score) as avg_score
                    FROM judgments j
                    JOIN conversations c ON j.conversation_id = c.id
                    WHERE c.simulation_run_id = %s AND j.judge_type = 'completeness'""",
                    (run.id,)
                )
                row = cursor.fetchone()
                avg_completeness = row["avg_score"] if row else None

            # Check total judgments for this run
            cursor.execute(
                """SELECT COUNT(*) as count FROM judgments j
                   JOIN conversations c ON j.conversation_id = c.id
                   WHERE c.simulation_run_id = %s""",
                (run.id,)
            )
            total_judgments = cursor.fetchone()["count"]

            run_data = {
                "id": run.id,
                "version": run.version,
                "status": run.status,
                "conversation_count": conv_count,
                "judged_count": judged_count,
                "target_count": target_count,
                "safety_pass_rate": safety_pass_rate,
                "avg_quality_score": avg_quality_score,
                "avg_completeness": avg_completeness,
                "has_judgments": total_judgments > 0,
                "started_at": run.started_at.strftime('%Y-%m-%d %H:%M') if run.started_at else None,
                "completed_at": run.completed_at.strftime('%Y-%m-%d %H:%M') if run.completed_at else None,
            }

            # Add conversation details if requested for this run
            if str(run.id) in expanded_run_ids and run.status in ("running", "judging"):
                run_data["conversations"] = get_conversations_for_run_status(run.id, run.status)

            result.append(run_data)

    return jsonify(result)


@app.route("/api/runs/<int:run_id>/judge", methods=["POST"])
@login_required
def run_judges_on_run(run_id: int):
    """Run judges on all conversations in a simulation run."""
    if not get_db_available():
        return jsonify({"error": "Database not available"}), 500

    from eval.database import get_simulation_run, update_simulation_run_status

    run = get_simulation_run(run_id)
    if not run:
        return jsonify({"error": "Run not found"}), 404

    # Update status to judging and run in background
    update_simulation_run_status(run_id, "judging", run.summary)

    import threading
    judge_model = run.config.get("judge_model", "gpt-4.1") if run.config else "gpt-4.1"

    def run_judges_background():
        from eval.database import init_pool, update_simulation_run_status
        init_pool()
        try:
            run_judges_on_simulation(run_id, judge_model)
            update_simulation_run_status(run_id, "completed", run.summary)
        except Exception as e:
            print(f"Error running judges: {e}")
            update_simulation_run_status(run_id, "completed", run.summary)

    thread = threading.Thread(target=run_judges_background, daemon=True)
    thread.start()

    return jsonify({
        "success": True,
        "status": "judging"
    })


@app.route("/")
@login_required
def index():
    """Redirect to runs page."""
    from flask import redirect
    return redirect("/runs")


@app.route("/api/version/<version_id>/status")
@login_required
def get_version_status(version_id: str):
    """Get the current status of a version/run for polling."""
    if not get_db_available():
        return jsonify({"error": "Database not available"}), 500

    from eval.database import get_cursor

    with get_cursor() as cursor:
        # Get run info
        cursor.execute(
            "SELECT id, status, config FROM simulation_runs WHERE version = %s",
            (version_id,)
        )
        run = cursor.fetchone()
        if not run:
            return jsonify({"error": "Run not found"}), 404

        run_id = run["id"]
        target_count = run["config"].get("count", 10) if run["config"] else 10

        # Get conversation count
        cursor.execute(
            "SELECT COUNT(*) as count FROM conversations WHERE simulation_run_id = %s",
            (run_id,)
        )
        conv_count = cursor.fetchone()["count"]

        # Get judged count
        cursor.execute(
            """SELECT COUNT(DISTINCT j.conversation_id) as count
               FROM judgments j
               JOIN conversations c ON j.conversation_id = c.id
               WHERE c.simulation_run_id = %s""",
            (run_id,)
        )
        judged_count = cursor.fetchone()["count"]

        # Get metrics if completed
        safety_pass_rate = None
        avg_quality_score = None
        avg_completeness = None

        if run["status"] == "completed" and conv_count > 0:
            cursor.execute(
                """SELECT
                    COUNT(CASE WHEN j.verdict = 'pass' THEN 1 END)::float /
                    NULLIF(COUNT(*), 0) as pass_rate
                FROM judgments j
                JOIN conversations c ON j.conversation_id = c.id
                WHERE c.simulation_run_id = %s AND j.judge_type = 'safety'""",
                (run_id,)
            )
            result = cursor.fetchone()
            safety_pass_rate = result["pass_rate"] if result else None

            cursor.execute(
                """SELECT AVG(j.score) as avg_score
                FROM judgments j
                JOIN conversations c ON j.conversation_id = c.id
                WHERE c.simulation_run_id = %s AND j.judge_type = 'quality'""",
                (run_id,)
            )
            result = cursor.fetchone()
            avg_quality_score = result["avg_score"] if result else None

            cursor.execute(
                """SELECT AVG(j.score) as avg_score
                FROM judgments j
                JOIN conversations c ON j.conversation_id = c.id
                WHERE c.simulation_run_id = %s AND j.judge_type = 'completeness'""",
                (run_id,)
            )
            result = cursor.fetchone()
            avg_completeness = result["avg_score"] if result else None

        return jsonify({
            "status": run["status"],
            "conversation_count": conv_count,
            "target_count": target_count,
            "judged_count": judged_count,
            "safety_pass_rate": safety_pass_rate,
            "avg_quality_score": avg_quality_score,
            "avg_completeness": avg_completeness,
        })


@app.route("/api/version/<version_id>/summary")
@login_required
def generate_version_summary(version_id: str):
    """Generate an AI summary of the run's performance with recommendations."""
    if not get_db_available():
        return jsonify({"error": "Database not available"}), 500

    from eval.database import get_cursor

    # Get the simulation run
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT id, version, config FROM simulation_runs WHERE version = %s",
            (version_id,)
        )
        run = cursor.fetchone()
        if not run:
            return jsonify({"error": "Run not found"}), 404

        run_id = run["id"]

        # Get all judgments with their details
        cursor.execute(
            """SELECT
                j.judge_type,
                j.judge_id,
                j.verdict,
                j.score,
                j.reasoning,
                c.persona,
                c.completion_reason
            FROM judgments j
            JOIN conversations c ON j.conversation_id = c.id
            WHERE c.simulation_run_id = %s
            ORDER BY j.judge_type, j.judge_id""",
            (run_id,)
        )
        judgments = cursor.fetchall()

        if not judgments:
            return jsonify({"error": "No judgments found for this run"}), 404

        # Get conversation count
        cursor.execute(
            "SELECT COUNT(*) as count FROM conversations WHERE simulation_run_id = %s",
            (run_id,)
        )
        conv_count = cursor.fetchone()["count"]

    # Build summary data for the LLM
    summary_data = {
        "version": version_id,
        "conversation_count": conv_count,
        "judgments_by_type": {},
    }

    for j in judgments:
        jtype = j["judge_type"]
        jid = j["judge_id"]
        key = f"{jtype}:{jid}"

        if key not in summary_data["judgments_by_type"]:
            summary_data["judgments_by_type"][key] = {
                "type": jtype,
                "id": jid,
                "verdicts": [],
                "scores": [],
                "sample_reasoning": [],
            }

        summary_data["judgments_by_type"][key]["verdicts"].append(j["verdict"])
        if j["score"] is not None:
            summary_data["judgments_by_type"][key]["scores"].append(j["score"])
        if j["reasoning"] and len(summary_data["judgments_by_type"][key]["sample_reasoning"]) < 5:
            # Include persona context with reasoning for failures and partial
            if j["verdict"] in ("fail", "partial"):
                persona = j["persona"] or {}
                summary_data["judgments_by_type"][key]["sample_reasoning"].append({
                    "persona_name": persona.get("name", "Unknown"),
                    "language": persona.get("language", "Unknown"),
                    "legal_issue": persona.get("legal_issue", "Unknown"),
                    "completion_reason": j["completion_reason"],
                    "reasoning": j["reasoning"][:800],  # Truncate long reasoning
                    "verdict": j["verdict"],
                })

    # Compute aggregate stats
    for key, data in summary_data["judgments_by_type"].items():
        verdicts = data["verdicts"]
        data["total"] = len(verdicts)
        data["pass_count"] = verdicts.count("pass")
        data["fail_count"] = verdicts.count("fail")
        data["partial_count"] = verdicts.count("partial")
        data["pass_rate"] = data["pass_count"] / len(verdicts) if verdicts else 0
        data["avg_score"] = sum(data["scores"]) / len(data["scores"]) if data["scores"] else None
        # Remove raw lists to reduce payload
        del data["verdicts"]
        del data["scores"]

    # Generate AI summary
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        return jsonify({"error": "OPENAI_API_KEY not configured"}), 500

    from openai import OpenAI
    client = OpenAI(api_key=openai_key)

    prompt = f"""Analyze this evaluation run for "Casey", a legal intake agent for Open Door Legal.

Run "{version_id}" - {conv_count} conversations:

{json.dumps(summary_data["judgments_by_type"], indent=2)}

Be concise. Use bullet points. Reference specific personas when noting issues.

## Issues
- Specific problems found, citing persona names (2-3 bullets)

## Recommendations
- Guidance for prompt improvements to address the issues (2-3 bullets)"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3,
        )
        summary = response.choices[0].message.content

        # Save summary to database
        with get_cursor() as cursor:
            cursor.execute(
                "UPDATE simulation_runs SET ai_summary = %s WHERE version = %s",
                (summary, version_id)
            )

        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/version/<version_id>")
@login_required
def version_detail(version_id: str):
    """Detail page for a specific version."""
    version = get_version(version_id)
    if not version:
        return "Version not found", 404

    # Get run status and saved summary from database
    run_status = "completed"
    ai_summary = None
    if get_db_available():
        from eval.database import get_cursor
        with get_cursor() as cursor:
            cursor.execute(
                "SELECT status, ai_summary FROM simulation_runs WHERE version = %s",
                (version_id,)
            )
            result = cursor.fetchone()
            if result:
                run_status = result["status"]
                ai_summary = result["ai_summary"]

    return render_template("version.html", version=version, run_status=run_status, ai_summary=ai_summary)


@app.route("/conversation/<session_id>")
@login_required
def conversation_detail(session_id: str):
    """Conversation inspector."""
    conversation = get_conversation(session_id)
    if not conversation:
        return "Conversation not found", 404
    return render_template("conversation.html", conversation=conversation)


@app.route("/conversation-by-file")
@login_required
def conversation_by_file():
    """Conversation inspector by filepath."""
    filepath = request.args.get("path")
    if not filepath:
        return "No path provided", 400
    conversation = get_conversation_by_file(filepath)
    if not conversation:
        return "Conversation not found", 404
    return render_template("conversation.html", conversation=conversation)


@app.route("/api/run-judges", methods=["POST"])
@login_required
def run_judges_api():
    """Run judges on a conversation (HTMX endpoint)."""
    data = request.json
    filepath = data.get("filepath")

    if not filepath:
        return jsonify({"error": "No filepath provided"}), 400

    # Import judge utilities
    try:
        from dotenv import load_dotenv
        load_dotenv()

        from openai import OpenAI
        from webapp.utils import get_conversation_by_file
        from eval.judges.base import ConversationContext, ConversationTurn
        from eval.judges.safety import SafetyEvaluator
        from eval.judges.quality import QualityEvaluator
        from eval.judges.completeness import CompletenessEvaluator

        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            return jsonify({"error": "OPENAI_API_KEY not set"}), 500

        llm_client = OpenAI(api_key=openai_key)

        # Load conversation
        conv_data = get_conversation_by_file(filepath)
        if not conv_data:
            return jsonify({"error": "Conversation not found"}), 404

        # Convert to context
        turns = []
        for msg in conv_data.get("transcript", []):
            role = "assistant" if msg["role"] == "casey" else "user"
            turns.append(ConversationTurn(role=role, content=msg["content"]))

        context = ConversationContext(
            turns=turns,
            persona=conv_data.get("persona", {}),
        )

        # Run judges
        safety_eval = SafetyEvaluator(llm_client=llm_client, model="gpt-4.1")
        quality_eval = QualityEvaluator(llm_client=llm_client, model="gpt-4.1")
        completeness_eval = CompletenessEvaluator(llm_client=llm_client, model="gpt-4.1")

        safety_results = safety_eval.evaluate_all(context)
        quality_results = quality_eval.evaluate_all(context)
        completeness_result = completeness_eval.evaluate(context)

        return jsonify({
            "safety": [r.to_dict() for r in safety_results],
            "quality": [r.to_dict() for r in quality_results],
            "completeness": completeness_result.to_dict(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Bind to 0.0.0.0 for Docker container access
    app.run(debug=True, host="0.0.0.0", port=5001)
