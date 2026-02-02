"""
Database module for ODL evaluation system.

Provides connection pooling and CRUD operations for all tables.
"""

import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, Json


# Global connection pool
_connection_pool: Optional[pool.ThreadedConnectionPool] = None


def get_database_url() -> str:
    """Get database URL from environment."""
    return os.environ.get(
        "DATABASE_URL",
        "postgresql://odl:odl_dev@localhost:5432/odl_eval"
    )


def init_pool(min_connections: int = 1, max_connections: int = 10) -> None:
    """Initialize the connection pool."""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = pool.ThreadedConnectionPool(
            min_connections,
            max_connections,
            get_database_url()
        )


def get_pool() -> pool.ThreadedConnectionPool:
    """Get the connection pool, initializing if needed."""
    global _connection_pool
    if _connection_pool is None:
        init_pool()
    return _connection_pool


@contextmanager
def get_connection():
    """Get a connection from the pool."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


@contextmanager
def get_cursor(cursor_factory=RealDictCursor):
    """Get a cursor with automatic connection management."""
    with get_connection() as conn:
        cursor = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cursor
        finally:
            cursor.close()


def close_pool() -> None:
    """Close all connections in the pool."""
    global _connection_pool
    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None


# =============================================================================
# Prompt Versions
# =============================================================================

@dataclass
class PromptVersion:
    id: Optional[int] = None
    version: str = ""
    name: Optional[str] = None
    content: str = ""
    metadata: Dict[str, Any] = None
    is_active: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def create_prompt_version(
    version: str,
    content: str,
    name: Optional[str] = None,
    metadata: Optional[Dict] = None,
    is_active: bool = False
) -> int:
    """Create a new prompt version. Returns the new ID."""
    with get_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO prompt_versions (version, name, content, metadata, is_active)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (version, name, content, Json(metadata or {}), is_active)
        )
        return cursor.fetchone()["id"]


def get_prompt_version(version: str) -> Optional[PromptVersion]:
    """Get a prompt version by version string."""
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT * FROM prompt_versions WHERE version = %s",
            (version,)
        )
        row = cursor.fetchone()
        if row:
            return PromptVersion(**row)
        return None


def get_prompt_version_by_id(id: int) -> Optional[PromptVersion]:
    """Get a prompt version by ID."""
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT * FROM prompt_versions WHERE id = %s",
            (id,)
        )
        row = cursor.fetchone()
        if row:
            return PromptVersion(**row)
        return None


def list_prompt_versions(active_only: bool = False) -> List[PromptVersion]:
    """List all prompt versions."""
    with get_cursor() as cursor:
        if active_only:
            cursor.execute(
                "SELECT * FROM prompt_versions WHERE is_active = true ORDER BY created_at DESC"
            )
        else:
            cursor.execute(
                "SELECT * FROM prompt_versions ORDER BY created_at DESC"
            )
        return [PromptVersion(**row) for row in cursor.fetchall()]


def set_active_prompt_version(version: str) -> None:
    """Set a prompt version as active (deactivates all others)."""
    with get_cursor() as cursor:
        cursor.execute("UPDATE prompt_versions SET is_active = false")
        cursor.execute(
            "UPDATE prompt_versions SET is_active = true WHERE version = %s",
            (version,)
        )


def delete_prompt_version(id: int) -> None:
    """Delete a prompt version by ID."""
    with get_cursor() as cursor:
        cursor.execute(
            "DELETE FROM prompt_versions WHERE id = %s",
            (id,)
        )


# =============================================================================
# Simulation Runs
# =============================================================================

@dataclass
class SimulationRun:
    id: Optional[int] = None
    version: str = ""
    prompt_version_id: Optional[int] = None
    config: Dict[str, Any] = None
    status: str = "pending"
    summary: Optional[Dict[str, Any]] = None
    ai_summary: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.config is None:
            self.config = {}


def create_simulation_run(
    version: str,
    config: Optional[Dict] = None,
    prompt_version_id: Optional[int] = None,
    status: str = "pending"
) -> int:
    """Create a new simulation run. Returns the new ID."""
    with get_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO simulation_runs (version, config, prompt_version_id, status, started_at)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (version, Json(config or {}), prompt_version_id, status, datetime.now())
        )
        return cursor.fetchone()["id"]


def get_simulation_run(id: int) -> Optional[SimulationRun]:
    """Get a simulation run by ID."""
    with get_cursor() as cursor:
        cursor.execute("SELECT * FROM simulation_runs WHERE id = %s", (id,))
        row = cursor.fetchone()
        if row:
            return SimulationRun(**row)
        return None


def get_simulation_run_by_version(version: str) -> Optional[SimulationRun]:
    """Get a simulation run by version string."""
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT * FROM simulation_runs WHERE version = %s ORDER BY created_at DESC LIMIT 1",
            (version,)
        )
        row = cursor.fetchone()
        if row:
            return SimulationRun(**row)
        return None


def list_simulation_runs(limit: int = 50) -> List[SimulationRun]:
    """List simulation runs ordered by creation date."""
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT * FROM simulation_runs ORDER BY created_at DESC LIMIT %s",
            (limit,)
        )
        return [SimulationRun(**row) for row in cursor.fetchall()]


def update_simulation_run_status(
    id: int,
    status: str,
    summary: Optional[Dict] = None
) -> None:
    """Update simulation run status and optionally summary."""
    with get_cursor() as cursor:
        if summary is not None:
            cursor.execute(
                """
                UPDATE simulation_runs
                SET status = %s, summary = %s, completed_at = %s
                WHERE id = %s
                """,
                (status, Json(summary), datetime.now() if status in ("completed", "failed") else None, id)
            )
        else:
            cursor.execute(
                """
                UPDATE simulation_runs
                SET status = %s, completed_at = %s
                WHERE id = %s
                """,
                (status, datetime.now() if status in ("completed", "failed") else None, id)
            )


# =============================================================================
# Conversations
# =============================================================================

@dataclass
class Conversation:
    id: Optional[int] = None
    simulation_run_id: Optional[int] = None
    session_id: str = ""
    persona: Dict[str, Any] = None
    transcript: List[Dict[str, Any]] = None
    completion_reason: Optional[str] = None
    turn_count: Optional[int] = None
    duration_seconds: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.persona is None:
            self.persona = {}
        if self.transcript is None:
            self.transcript = []
        if self.metadata is None:
            self.metadata = {}


def create_conversation(
    session_id: str,
    persona: Dict,
    transcript: List[Dict],
    simulation_run_id: Optional[int] = None,
    completion_reason: Optional[str] = None,
    turn_count: Optional[int] = None,
    duration_seconds: Optional[float] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    metadata: Optional[Dict] = None
) -> int:
    """Create a new conversation. Returns the new ID."""
    with get_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO conversations (
                simulation_run_id, session_id, persona, transcript,
                completion_reason, turn_count, duration_seconds,
                start_time, end_time, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                simulation_run_id, session_id, Json(persona), Json(transcript),
                completion_reason, turn_count, duration_seconds,
                start_time, end_time, Json(metadata or {})
            )
        )
        return cursor.fetchone()["id"]


def get_conversation(id: int) -> Optional[Conversation]:
    """Get a conversation by ID."""
    with get_cursor() as cursor:
        cursor.execute("SELECT * FROM conversations WHERE id = %s", (id,))
        row = cursor.fetchone()
        if row:
            return Conversation(**row)
        return None


def get_conversation_by_session_id(session_id: str) -> Optional[Conversation]:
    """Get a conversation by session ID."""
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT * FROM conversations WHERE session_id = %s",
            (session_id,)
        )
        row = cursor.fetchone()
        if row:
            return Conversation(**row)
        return None


def list_conversations_by_simulation(simulation_run_id: int) -> List[Conversation]:
    """List all conversations for a simulation run."""
    with get_cursor() as cursor:
        cursor.execute(
            """
            SELECT * FROM conversations
            WHERE simulation_run_id = %s
            ORDER BY start_time
            """,
            (simulation_run_id,)
        )
        return [Conversation(**row) for row in cursor.fetchall()]


def list_conversations(limit: int = 100) -> List[Conversation]:
    """List recent conversations."""
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT * FROM conversations ORDER BY created_at DESC LIMIT %s",
            (limit,)
        )
        return [Conversation(**row) for row in cursor.fetchall()]


# =============================================================================
# Judge Configs
# =============================================================================

@dataclass
class JudgeConfig:
    id: Optional[int] = None
    judge_type: str = ""
    judge_id: str = ""
    version: str = ""
    config: Dict[str, Any] = None
    prompt_template: Optional[str] = None
    is_active: bool = True
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.config is None:
            self.config = {}


def create_judge_config(
    judge_type: str,
    judge_id: str,
    version: str,
    config: Optional[Dict] = None,
    prompt_template: Optional[str] = None,
    is_active: bool = True
) -> int:
    """Create a new judge config. Returns the new ID."""
    with get_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO judge_configs (judge_type, judge_id, version, config, prompt_template, is_active)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (judge_type, judge_id, version, Json(config or {}), prompt_template, is_active)
        )
        return cursor.fetchone()["id"]


def get_judge_config(judge_id: str, version: str) -> Optional[JudgeConfig]:
    """Get a judge config by judge_id and version."""
    with get_cursor() as cursor:
        cursor.execute(
            "SELECT * FROM judge_configs WHERE judge_id = %s AND version = %s",
            (judge_id, version)
        )
        row = cursor.fetchone()
        if row:
            return JudgeConfig(**row)
        return None


def list_judge_configs(active_only: bool = True) -> List[JudgeConfig]:
    """List judge configs."""
    with get_cursor() as cursor:
        if active_only:
            cursor.execute(
                "SELECT * FROM judge_configs WHERE is_active = true ORDER BY judge_type, judge_id"
            )
        else:
            cursor.execute(
                "SELECT * FROM judge_configs ORDER BY judge_type, judge_id"
            )
        return [JudgeConfig(**row) for row in cursor.fetchall()]


# =============================================================================
# Judgment Runs
# =============================================================================

@dataclass
class JudgmentRun:
    id: Optional[int] = None
    simulation_run_id: Optional[int] = None
    judge_config_snapshot: Optional[Dict[str, Any]] = None
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


def create_judgment_run(
    simulation_run_id: int,
    judge_config_snapshot: Optional[Dict] = None,
    status: str = "pending"
) -> int:
    """Create a new judgment run. Returns the new ID."""
    with get_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO judgment_runs (simulation_run_id, judge_config_snapshot, status, started_at)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (simulation_run_id, Json(judge_config_snapshot) if judge_config_snapshot else None, status, datetime.now())
        )
        return cursor.fetchone()["id"]


def get_judgment_run(id: int) -> Optional[JudgmentRun]:
    """Get a judgment run by ID."""
    with get_cursor() as cursor:
        cursor.execute("SELECT * FROM judgment_runs WHERE id = %s", (id,))
        row = cursor.fetchone()
        if row:
            return JudgmentRun(**row)
        return None


def update_judgment_run_status(id: int, status: str) -> None:
    """Update judgment run status."""
    with get_cursor() as cursor:
        cursor.execute(
            """
            UPDATE judgment_runs
            SET status = %s, completed_at = %s
            WHERE id = %s
            """,
            (status, datetime.now() if status in ("completed", "failed") else None, id)
        )


# =============================================================================
# Judgments
# =============================================================================

@dataclass
class Judgment:
    id: Optional[int] = None
    judgment_run_id: Optional[int] = None
    conversation_id: Optional[int] = None
    judge_type: str = ""
    judge_id: str = ""
    verdict: Optional[str] = None
    score: Optional[float] = None
    reasoning: Optional[str] = None
    evidence: Optional[List[str]] = None
    metadata: Dict[str, Any] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def create_judgment(
    conversation_id: int,
    judge_type: str,
    judge_id: str,
    verdict: Optional[str] = None,
    score: Optional[float] = None,
    reasoning: Optional[str] = None,
    evidence: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
    judgment_run_id: Optional[int] = None
) -> int:
    """Create a new judgment. Returns the new ID."""
    with get_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO judgments (
                judgment_run_id, conversation_id, judge_type, judge_id,
                verdict, score, reasoning, evidence, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                judgment_run_id, conversation_id, judge_type, judge_id,
                verdict, score, reasoning, Json(evidence) if evidence else None, Json(metadata or {})
            )
        )
        return cursor.fetchone()["id"]


def get_judgments_for_conversation(conversation_id: int) -> List[Judgment]:
    """Get all judgments for a conversation."""
    with get_cursor() as cursor:
        cursor.execute(
            """
            SELECT * FROM judgments
            WHERE conversation_id = %s
            ORDER BY judge_type, judge_id
            """,
            (conversation_id,)
        )
        return [Judgment(**row) for row in cursor.fetchall()]


def get_judgments_for_run(judgment_run_id: int) -> List[Judgment]:
    """Get all judgments for a judgment run."""
    with get_cursor() as cursor:
        cursor.execute(
            """
            SELECT * FROM judgments
            WHERE judgment_run_id = %s
            ORDER BY conversation_id, judge_type, judge_id
            """,
            (judgment_run_id,)
        )
        return [Judgment(**row) for row in cursor.fetchall()]


# =============================================================================
# Aggregate Queries
# =============================================================================

def get_version_summary(version: str) -> Optional[Dict[str, Any]]:
    """Get summary statistics for a version."""
    with get_cursor() as cursor:
        # Get simulation run
        cursor.execute(
            """
            SELECT * FROM simulation_runs WHERE version = %s
            ORDER BY created_at DESC LIMIT 1
            """,
            (version,)
        )
        run = cursor.fetchone()
        if not run:
            return None

        # Get conversation stats
        cursor.execute(
            """
            SELECT
                COUNT(*) as conversation_count,
                AVG(turn_count) as avg_turns,
                AVG(duration_seconds) as avg_duration,
                COUNT(*) FILTER (WHERE completion_reason = 'intake_complete') as completed_count
            FROM conversations
            WHERE simulation_run_id = %s
            """,
            (run["id"],)
        )
        conv_stats = cursor.fetchone()

        # Get judgment stats
        cursor.execute(
            """
            SELECT
                j.judge_type,
                j.judge_id,
                COUNT(*) FILTER (WHERE j.verdict = 'pass') as pass_count,
                COUNT(*) FILTER (WHERE j.verdict = 'fail') as fail_count,
                AVG(j.score) as avg_score
            FROM judgments j
            JOIN conversations c ON j.conversation_id = c.id
            WHERE c.simulation_run_id = %s
            GROUP BY j.judge_type, j.judge_id
            """,
            (run["id"],)
        )
        judgment_stats = cursor.fetchall()

        return {
            "version": version,
            "simulation_run_id": run["id"],
            "status": run["status"],
            "started_at": run["started_at"],
            "completed_at": run["completed_at"],
            "conversation_count": conv_stats["conversation_count"],
            "avg_turns": float(conv_stats["avg_turns"]) if conv_stats["avg_turns"] else None,
            "avg_duration": float(conv_stats["avg_duration"]) if conv_stats["avg_duration"] else None,
            "completion_rate": (
                conv_stats["completed_count"] / conv_stats["conversation_count"]
                if conv_stats["conversation_count"] > 0 else 0
            ),
            "judgment_stats": judgment_stats,
        }


def get_conversations_with_judgments(simulation_run_id: int) -> List[Dict[str, Any]]:
    """Get all conversations with their judgments for a simulation run."""
    with get_cursor() as cursor:
        cursor.execute(
            """
            SELECT
                c.*,
                COALESCE(
                    json_agg(
                        json_build_object(
                            'judge_type', j.judge_type,
                            'judge_id', j.judge_id,
                            'verdict', j.verdict,
                            'score', j.score,
                            'reasoning', j.reasoning,
                            'evidence', j.evidence,
                            'metadata', j.metadata
                        )
                    ) FILTER (WHERE j.id IS NOT NULL),
                    '[]'
                ) as judgments
            FROM conversations c
            LEFT JOIN judgments j ON j.conversation_id = c.id
            WHERE c.simulation_run_id = %s
            GROUP BY c.id
            ORDER BY c.start_time
            """,
            (simulation_run_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
