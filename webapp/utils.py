"""
Data loading utilities for the eval dashboard.

Supports both database (PostgreSQL) and filesystem (JSON files) backends.
The database backend is preferred when available.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List

# Check if database is available
_use_database = None


def _should_use_database() -> bool:
    """Check if we should use the database backend."""
    global _use_database
    if _use_database is not None:
        return _use_database

    # Check environment variable override
    if os.environ.get("USE_FILESYSTEM", "").lower() in ("true", "1", "yes"):
        _use_database = False
        return False

    # Try to import and check database
    try:
        from webapp.db import is_database_available, init_pool
        from eval.database import init_pool as db_init_pool
        db_init_pool()
        _use_database = is_database_available()
    except Exception as e:
        print(f"Database not available, using filesystem: {e}")
        _use_database = False

    return _use_database


@dataclass
class ConversationSummary:
    """Summary of a single conversation."""
    session_id: str
    persona_name: str
    language: str
    legal_issue: str
    turn_count: int
    duration_seconds: float
    completion_reason: str
    version: Optional[str]
    start_time: str
    filepath: str
    has_eval: bool = False
    safety_passed: Optional[bool] = None
    quality_score: Optional[float] = None
    intake_completeness: Optional[float] = None  # 0-1 completion rate
    intake_steps_completed: Optional[int] = None
    intake_steps_total: Optional[int] = None


@dataclass
class VersionSummary:
    """Summary of all conversations for a version."""
    version: str
    start_time: str
    conversation_count: int
    completion_rate: float
    safety_pass_rate: Optional[float]
    avg_quality_score: Optional[float]
    avg_intake_completeness: Optional[float]  # Average intake completion rate
    completion_reasons: dict
    conversations: List[ConversationSummary]


def load_transcript(filepath: Path) -> dict:
    """Load a transcript JSON file."""
    return json.loads(filepath.read_text())


def load_eval_results(filepath: Path) -> Optional[dict]:
    """Load eval results for a transcript if they exist."""
    eval_path = filepath.parent / "evals" / f"{filepath.stem}_eval.json"
    if eval_path.exists():
        return json.loads(eval_path.read_text())
    return None


def compute_eval_metrics(eval_data: dict) -> tuple[bool, float, float, int, int]:
    """
    Compute safety pass, quality score, and intake completeness from eval results.
    Returns (safety_passed, avg_quality_score, intake_completeness, steps_completed, steps_total).
    """
    # Safety: pass if no failures
    safety_results = eval_data.get("safety", [])
    safety_passed = all(
        r.get("verdict") != "fail" for r in safety_results
    )

    # Quality: average of non-null scores
    quality_results = eval_data.get("quality", [])
    scores = [r.get("score") for r in quality_results if r.get("score") is not None]
    avg_quality = sum(scores) / len(scores) if scores else None

    # Completeness: get completion rate from completeness judge
    completeness_data = eval_data.get("completeness", {})
    completeness_metadata = completeness_data.get("metadata", {}) if completeness_data else {}
    intake_completeness = completeness_metadata.get("completion_rate")
    steps_completed = completeness_metadata.get("steps_completed")
    steps_total = completeness_metadata.get("steps_total")

    return safety_passed, avg_quality, intake_completeness, steps_completed, steps_total


def get_transcripts_dir() -> Path:
    """Get the transcripts directory."""
    # Try relative to webapp, then relative to project root
    candidates = [
        Path(__file__).parent.parent / "transcripts",
        Path("transcripts"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def list_versions(transcripts_dir: Path = None) -> List[VersionSummary]:
    """List all versions with their summaries."""
    # Try database first
    if _should_use_database():
        try:
            from webapp.db import list_versions_from_db
            db_versions = list_versions_from_db()
            # Convert db dataclasses to local dataclasses for compatibility
            return [
                VersionSummary(
                    version=v.version,
                    start_time=v.start_time,
                    conversation_count=v.conversation_count,
                    completion_rate=v.completion_rate,
                    safety_pass_rate=v.safety_pass_rate,
                    avg_quality_score=v.avg_quality_score,
                    avg_intake_completeness=v.avg_intake_completeness,
                    completion_reasons=v.completion_reasons,
                    conversations=[
                        ConversationSummary(
                            session_id=c.session_id,
                            persona_name=c.persona_name,
                            language=c.language,
                            legal_issue=c.legal_issue,
                            turn_count=c.turn_count,
                            duration_seconds=c.duration_seconds,
                            completion_reason=c.completion_reason,
                            version=c.version,
                            start_time=c.start_time,
                            filepath=c.filepath,
                            has_eval=c.has_eval,
                            safety_passed=c.safety_passed,
                            quality_score=c.quality_score,
                            intake_completeness=c.intake_completeness,
                            intake_steps_completed=c.intake_steps_completed,
                            intake_steps_total=c.intake_steps_total,
                        )
                        for c in v.conversations
                    ]
                )
                for v in db_versions
            ]
        except Exception as e:
            print(f"Database error, falling back to filesystem: {e}")

    # Fall back to filesystem
    return _list_versions_from_filesystem(transcripts_dir)


def _list_versions_from_filesystem(transcripts_dir: Path = None) -> List[VersionSummary]:
    """List all versions from filesystem."""
    if transcripts_dir is None:
        transcripts_dir = get_transcripts_dir()

    if not transcripts_dir.exists():
        return []

    # Group conversations by version
    versions = {}

    # Load batch summaries first
    for summary_file in transcripts_dir.glob("batch_summary_*.json"):
        data = json.loads(summary_file.read_text())
        version = data.get("version") or summary_file.stem.replace("batch_summary_", "")
        if version not in versions:
            versions[version] = {
                "start_time": data.get("start_time"),
                "completion_reasons": data.get("completion_reasons", {}),
                "conversations": [],
            }

    # Load individual transcripts
    for transcript_file in transcripts_dir.glob("*.json"):
        if transcript_file.name.startswith("batch_summary"):
            continue

        try:
            data = json.loads(transcript_file.read_text())
        except json.JSONDecodeError:
            continue

        version = data.get("version") or "unversioned"
        if version not in versions:
            versions[version] = {
                "start_time": data.get("start_time"),
                "completion_reasons": {},
                "conversations": [],
            }

        persona = data.get("persona", {})

        # Load eval results if available
        eval_data = load_eval_results(transcript_file)
        has_eval = eval_data is not None
        safety_passed = None
        quality_score = None
        intake_completeness = None
        intake_steps_completed = None
        intake_steps_total = None
        if eval_data:
            safety_passed, quality_score, intake_completeness, intake_steps_completed, intake_steps_total = compute_eval_metrics(eval_data)

        conv = ConversationSummary(
            session_id=data.get("session_id", transcript_file.stem),
            persona_name=persona.get("name", "Unknown"),
            language=persona.get("language", "Unknown"),
            legal_issue=persona.get("legal_issue", "Unknown"),
            turn_count=data.get("turn_count", 0),
            duration_seconds=data.get("duration_seconds", 0),
            completion_reason=data.get("completion_reason", "unknown"),
            version=version,
            start_time=data.get("start_time", ""),
            filepath=str(transcript_file),
            has_eval=has_eval,
            safety_passed=safety_passed,
            quality_score=quality_score,
            intake_completeness=intake_completeness,
            intake_steps_completed=intake_steps_completed,
            intake_steps_total=intake_steps_total,
        )
        versions[version]["conversations"].append(conv)

    # Build version summaries
    result = []
    for version, info in versions.items():
        convs = info["conversations"]
        if not convs:
            continue

        completed = sum(1 for c in convs if c.completion_reason == "intake_complete")
        completion_rate = completed / len(convs) if convs else 0

        # Aggregate eval metrics
        convs_with_eval = [c for c in convs if c.has_eval]
        safety_pass_rate = None
        avg_quality_score = None

        avg_intake_completeness = None

        if convs_with_eval:
            # Safety pass rate: % of evaluated conversations that passed all safety checks
            safety_passed_count = sum(1 for c in convs_with_eval if c.safety_passed)
            safety_pass_rate = safety_passed_count / len(convs_with_eval)

            # Avg quality score across all evaluated conversations
            quality_scores = [c.quality_score for c in convs_with_eval if c.quality_score is not None]
            if quality_scores:
                avg_quality_score = sum(quality_scores) / len(quality_scores)

            # Avg intake completeness across all evaluated conversations
            completeness_scores = [c.intake_completeness for c in convs_with_eval if c.intake_completeness is not None]
            if completeness_scores:
                avg_intake_completeness = sum(completeness_scores) / len(completeness_scores)

        result.append(VersionSummary(
            version=version,
            start_time=info.get("start_time") or (convs[0].start_time if convs else ""),
            conversation_count=len(convs),
            completion_rate=completion_rate,
            safety_pass_rate=safety_pass_rate,
            avg_quality_score=avg_quality_score,
            avg_intake_completeness=avg_intake_completeness,
            completion_reasons=info.get("completion_reasons", {}),
            conversations=convs,
        ))

    # Sort by start time (newest first)
    result.sort(key=lambda v: v.start_time or "", reverse=True)
    return result


def get_version(version_id: str, transcripts_dir: Path = None) -> Optional[VersionSummary]:
    """Get a specific version's summary."""
    versions = list_versions(transcripts_dir)
    for v in versions:
        if v.version == version_id:
            return v
    return None


def get_conversation(session_id: str, transcripts_dir: Path = None) -> Optional[dict]:
    """Get a specific conversation by session ID."""
    # Try database first
    if _should_use_database():
        try:
            from webapp.db import get_conversation_from_db
            result = get_conversation_from_db(session_id)
            if result:
                return result
        except Exception as e:
            print(f"Database error, falling back to filesystem: {e}")

    # Fall back to filesystem
    return _get_conversation_from_filesystem(session_id, transcripts_dir)


def _get_conversation_from_filesystem(session_id: str, transcripts_dir: Path = None) -> Optional[dict]:
    """Get a specific conversation from filesystem."""
    if transcripts_dir is None:
        transcripts_dir = get_transcripts_dir()

    for transcript_file in transcripts_dir.glob("*.json"):
        if transcript_file.name.startswith("batch_summary"):
            continue

        try:
            data = json.loads(transcript_file.read_text())
            if data.get("session_id") == session_id:
                data["_filepath"] = str(transcript_file)
                # Load eval results if they exist
                eval_data = load_eval_results(transcript_file)
                if eval_data:
                    data["_eval"] = eval_data
                return data
        except json.JSONDecodeError:
            continue

    return None


def get_conversation_by_file(filepath: str) -> Optional[dict]:
    """Get a conversation by its filepath."""
    # Check if this is a database reference
    if filepath.startswith("db://conversations/"):
        if _should_use_database():
            try:
                from webapp.db import get_conversation_by_id_from_db
                conversation_id = int(filepath.split("/")[-1])
                return get_conversation_by_id_from_db(conversation_id)
            except Exception as e:
                print(f"Database error: {e}")
                return None
        return None

    # Filesystem path
    path = Path(filepath)
    if path.exists():
        data = json.loads(path.read_text())
        data["_filepath"] = str(path)
        # Load eval results if they exist
        eval_data = load_eval_results(path)
        if eval_data:
            data["_eval"] = eval_data
        return data
    return None
