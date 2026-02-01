"""
Database utilities for the webapp.

This module provides database query functions specifically for the webapp,
building on the core database module.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.database import (
    init_pool,
    close_pool,
    get_cursor,
    get_connection,
    SimulationRun,
    Conversation,
    Judgment,
    list_simulation_runs,
    get_simulation_run_by_version,
    get_conversation_by_session_id,
    get_judgments_for_conversation,
    get_conversations_with_judgments,
)
from psycopg2.extras import RealDictCursor


def is_database_available() -> bool:
    """Check if database is available and has data."""
    try:
        with get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM conversations")
            count = cursor.fetchone()["count"]
            return count > 0
    except Exception:
        return False


@dataclass
class ConversationSummary:
    """Summary of a single conversation for display."""
    session_id: str
    persona_name: str
    language: str
    legal_issue: str
    turn_count: int
    duration_seconds: float
    completion_reason: str
    version: Optional[str]
    start_time: str
    filepath: str  # For backwards compatibility
    has_eval: bool = False
    safety_passed: Optional[bool] = None
    quality_score: Optional[float] = None
    intake_completeness: Optional[float] = None
    intake_steps_completed: Optional[int] = None
    intake_steps_total: Optional[int] = None
    conversation_id: Optional[int] = None


@dataclass
class VersionSummary:
    """Summary of all conversations for a version."""
    version: str
    start_time: str
    conversation_count: int
    completion_rate: float
    safety_pass_rate: Optional[float]
    avg_quality_score: Optional[float]
    avg_intake_completeness: Optional[float]
    completion_reasons: dict
    conversations: List[ConversationSummary]
    simulation_run_id: Optional[int] = None


def list_versions_from_db() -> List[VersionSummary]:
    """List all versions with their summaries from the database."""
    with get_cursor() as cursor:
        # Get all simulation runs
        cursor.execute("""
            SELECT
                sr.id,
                sr.version,
                sr.started_at,
                sr.status,
                COUNT(c.id) as conversation_count,
                COUNT(c.id) FILTER (WHERE c.completion_reason = 'intake_complete') as completed_count
            FROM simulation_runs sr
            LEFT JOIN conversations c ON c.simulation_run_id = sr.id
            GROUP BY sr.id
            ORDER BY sr.started_at DESC
        """)
        runs = cursor.fetchall()

        versions = []
        for run in runs:
            # Get conversations for this run
            cursor.execute("""
                SELECT
                    c.id,
                    c.session_id,
                    c.persona,
                    c.turn_count,
                    c.duration_seconds,
                    c.completion_reason,
                    c.start_time
                FROM conversations c
                WHERE c.simulation_run_id = %s
                ORDER BY c.start_time
            """, (run["id"],))
            conv_rows = cursor.fetchall()

            conversations = []
            safety_passed_count = 0
            quality_scores = []
            completeness_scores = []
            convs_with_eval = 0

            for conv_row in conv_rows:
                persona = conv_row["persona"] or {}

                # Get judgments for this conversation
                cursor.execute("""
                    SELECT judge_type, judge_id, verdict, score, metadata
                    FROM judgments
                    WHERE conversation_id = %s
                """, (conv_row["id"],))
                judgments = cursor.fetchall()

                has_eval = len(judgments) > 0
                safety_passed = None
                quality_score = None
                intake_completeness = None
                intake_steps_completed = None
                intake_steps_total = None

                if has_eval:
                    convs_with_eval += 1

                    # Check safety (all must pass or be N/A)
                    safety_results = [j for j in judgments if j["judge_type"] == "safety"]
                    if safety_results:
                        safety_passed = all(
                            j["verdict"] != "fail" for j in safety_results
                        )
                        if safety_passed:
                            safety_passed_count += 1

                    # Average quality score
                    quality_results = [j for j in judgments if j["judge_type"] == "quality" and j["score"] is not None]
                    if quality_results:
                        quality_score = sum(j["score"] for j in quality_results) / len(quality_results)
                        quality_scores.append(quality_score)

                    # Completeness
                    completeness_results = [j for j in judgments if j["judge_type"] == "completeness"]
                    if completeness_results:
                        comp = completeness_results[0]
                        metadata = comp["metadata"] or {}
                        intake_completeness = metadata.get("completion_rate")
                        intake_steps_completed = metadata.get("steps_completed")
                        intake_steps_total = metadata.get("steps_total")
                        if intake_completeness is not None:
                            completeness_scores.append(intake_completeness)

                conv_summary = ConversationSummary(
                    session_id=conv_row["session_id"],
                    persona_name=persona.get("name", "Unknown"),
                    language=persona.get("language", "Unknown"),
                    legal_issue=persona.get("legal_issue", "Unknown"),
                    turn_count=conv_row["turn_count"] or 0,
                    duration_seconds=conv_row["duration_seconds"] or 0,
                    completion_reason=conv_row["completion_reason"] or "unknown",
                    version=run["version"],
                    start_time=conv_row["start_time"].isoformat() if conv_row["start_time"] else "",
                    filepath=f"db://conversations/{conv_row['id']}",
                    has_eval=has_eval,
                    safety_passed=safety_passed,
                    quality_score=quality_score,
                    intake_completeness=intake_completeness,
                    intake_steps_completed=intake_steps_completed,
                    intake_steps_total=intake_steps_total,
                    conversation_id=conv_row["id"],
                )
                conversations.append(conv_summary)

            # Calculate aggregate stats
            safety_pass_rate = (
                safety_passed_count / convs_with_eval
                if convs_with_eval > 0 else None
            )
            avg_quality_score = (
                sum(quality_scores) / len(quality_scores)
                if quality_scores else None
            )
            avg_intake_completeness = (
                sum(completeness_scores) / len(completeness_scores)
                if completeness_scores else None
            )
            completion_rate = (
                run["completed_count"] / run["conversation_count"]
                if run["conversation_count"] > 0 else 0
            )

            # Build completion reasons dict
            completion_reasons = {}
            for conv in conversations:
                reason = conv.completion_reason
                completion_reasons[reason] = completion_reasons.get(reason, 0) + 1

            version_summary = VersionSummary(
                version=run["version"],
                start_time=run["started_at"].isoformat() if run["started_at"] else "",
                conversation_count=run["conversation_count"],
                completion_rate=completion_rate,
                safety_pass_rate=safety_pass_rate,
                avg_quality_score=avg_quality_score,
                avg_intake_completeness=avg_intake_completeness,
                completion_reasons=completion_reasons,
                conversations=conversations,
                simulation_run_id=run["id"],
            )
            versions.append(version_summary)

        return versions


def get_version_from_db(version_id: str) -> Optional[VersionSummary]:
    """Get a specific version's summary from the database."""
    versions = list_versions_from_db()
    for v in versions:
        if v.version == version_id:
            return v
    return None


def get_conversation_from_db(session_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific conversation by session ID from the database."""
    with get_cursor() as cursor:
        cursor.execute("""
            SELECT c.*, sr.version
            FROM conversations c
            LEFT JOIN simulation_runs sr ON c.simulation_run_id = sr.id
            WHERE c.session_id = %s
        """, (session_id,))
        row = cursor.fetchone()

        if not row:
            return None

        # Get judgments
        cursor.execute("""
            SELECT judge_type, judge_id, verdict, score, reasoning, evidence, metadata
            FROM judgments
            WHERE conversation_id = %s
            ORDER BY judge_type, judge_id
        """, (row["id"],))
        judgments = cursor.fetchall()

        # Build eval dict from judgments
        eval_data = None
        if judgments:
            eval_data = {
                "safety": [],
                "quality": [],
                "completeness": None,
            }
            for j in judgments:
                judgment_dict = {
                    "judge_id": j["judge_id"],
                    "verdict": j["verdict"],
                    "score": j["score"],
                    "reasoning": j["reasoning"],
                    "evidence": j["evidence"],
                    "metadata": j["metadata"],
                }
                if j["judge_type"] == "safety":
                    eval_data["safety"].append(judgment_dict)
                elif j["judge_type"] == "quality":
                    eval_data["quality"].append(judgment_dict)
                elif j["judge_type"] == "completeness":
                    eval_data["completeness"] = judgment_dict

        result = {
            "version": row["version"],
            "persona": row["persona"],
            "session_id": row["session_id"],
            "start_time": row["start_time"].isoformat() if row["start_time"] else None,
            "end_time": row["end_time"].isoformat() if row["end_time"] else None,
            "duration_seconds": row["duration_seconds"],
            "turn_count": row["turn_count"],
            "completion_reason": row["completion_reason"],
            "transcript": row["transcript"],
            "_filepath": f"db://conversations/{row['id']}",
            "_conversation_id": row["id"],
        }

        if eval_data:
            result["_eval"] = eval_data

        return result


def get_conversation_by_id_from_db(conversation_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific conversation by database ID."""
    with get_cursor() as cursor:
        cursor.execute("""
            SELECT c.*, sr.version
            FROM conversations c
            LEFT JOIN simulation_runs sr ON c.simulation_run_id = sr.id
            WHERE c.id = %s
        """, (conversation_id,))
        row = cursor.fetchone()

        if not row:
            return None

        # Get judgments
        cursor.execute("""
            SELECT judge_type, judge_id, verdict, score, reasoning, evidence, metadata
            FROM judgments
            WHERE conversation_id = %s
            ORDER BY judge_type, judge_id
        """, (conversation_id,))
        judgments = cursor.fetchall()

        # Build eval dict from judgments
        eval_data = None
        if judgments:
            eval_data = {
                "safety": [],
                "quality": [],
                "completeness": None,
            }
            for j in judgments:
                judgment_dict = {
                    "judge_id": j["judge_id"],
                    "verdict": j["verdict"],
                    "score": j["score"],
                    "reasoning": j["reasoning"],
                    "evidence": j["evidence"],
                    "metadata": j["metadata"],
                }
                if j["judge_type"] == "safety":
                    eval_data["safety"].append(judgment_dict)
                elif j["judge_type"] == "quality":
                    eval_data["quality"].append(judgment_dict)
                elif j["judge_type"] == "completeness":
                    eval_data["completeness"] = judgment_dict

        result = {
            "version": row["version"],
            "persona": row["persona"],
            "session_id": row["session_id"],
            "start_time": row["start_time"].isoformat() if row["start_time"] else None,
            "end_time": row["end_time"].isoformat() if row["end_time"] else None,
            "duration_seconds": row["duration_seconds"],
            "turn_count": row["turn_count"],
            "completion_reason": row["completion_reason"],
            "transcript": row["transcript"],
            "_filepath": f"db://conversations/{row['id']}",
            "_conversation_id": row["id"],
        }

        if eval_data:
            result["_eval"] = eval_data

        return result
