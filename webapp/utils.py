"""
Data loading utilities for the eval dashboard.
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional


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


@dataclass
class VersionSummary:
    """Summary of all conversations for a version."""
    version: str
    start_time: str
    conversation_count: int
    completion_rate: float
    safety_pass_rate: Optional[float]
    avg_quality_score: Optional[float]
    completion_reasons: dict
    conversations: list[ConversationSummary]


def load_transcript(filepath: Path) -> dict:
    """Load a transcript JSON file."""
    return json.loads(filepath.read_text())


def load_eval_results(filepath: Path) -> Optional[dict]:
    """Load eval results for a transcript if they exist."""
    eval_path = filepath.parent / "evals" / f"{filepath.stem}_eval.json"
    if eval_path.exists():
        return json.loads(eval_path.read_text())
    return None


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


def list_versions(transcripts_dir: Path = None) -> list[VersionSummary]:
    """List all versions with their summaries."""
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

        result.append(VersionSummary(
            version=version,
            start_time=info.get("start_time") or (convs[0].start_time if convs else ""),
            conversation_count=len(convs),
            completion_rate=completion_rate,
            safety_pass_rate=None,  # Populated after running judges
            avg_quality_score=None,
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
    if transcripts_dir is None:
        transcripts_dir = get_transcripts_dir()

    for transcript_file in transcripts_dir.glob("*.json"):
        if transcript_file.name.startswith("batch_summary"):
            continue

        try:
            data = json.loads(transcript_file.read_text())
            if data.get("session_id") == session_id:
                data["_filepath"] = str(transcript_file)
                return data
        except json.JSONDecodeError:
            continue

    return None


def get_conversation_by_file(filepath: str) -> Optional[dict]:
    """Get a conversation by its filepath."""
    path = Path(filepath)
    if path.exists():
        data = json.loads(path.read_text())
        data["_filepath"] = str(path)
        return data
    return None
