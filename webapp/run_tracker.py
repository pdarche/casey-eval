"""
In-memory tracker for active evaluation runs.

Tracks conversation progress before data is written to the database,
enabling real-time visibility into running conversations.
"""

import threading
from typing import Any

# Thread-safe lock for all operations
_lock = threading.Lock()

# Active runs storage
# Structure:
# {
#     run_id: {
#         "personas": [...],              # Full persona list
#         "active": {index: {...}},       # Currently running conversations
#         "completed_indices": set(),     # Finished conversation indices
#     }
# }
_active_runs: dict[int, dict[str, Any]] = {}

# Cancelled runs — checked cooperatively by background threads
_cancelled_runs: set[int] = set()

# Per-conversation judging status: conversation_id -> "judging" | "done" | "error"
_judging_conversations: dict[int, str] = {}


def register_run_generating(run_id: int, count: int) -> None:
    """
    Register a new run as generating personas.

    Called at the start of a run, before persona generation begins.

    Args:
        run_id: The simulation run ID
        count: The requested number of personas
    """
    with _lock:
        _active_runs[run_id] = {
            "phase": "generating",
            "target_count": count,
            "personas": [],
            "active": {},
            "completed_indices": set(),
        }


def register_run(run_id: int, personas: list[Any]) -> None:
    """
    Register a run's persona list and transition to conversation phase.

    Called after persona generation completes, before conversations begin.

    Args:
        run_id: The simulation run ID
        personas: List of persona objects/dicts for this run
    """
    with _lock:
        if run_id in _active_runs:
            _active_runs[run_id]["phase"] = "running"
            _active_runs[run_id]["personas"] = [_persona_to_dict(p) for p in personas]
        else:
            _active_runs[run_id] = {
                "phase": "running",
                "personas": [_persona_to_dict(p) for p in personas],
                "active": {},
                "completed_indices": set(),
            }


def start_conversation(run_id: int, index: int, persona: Any) -> None:
    """
    Mark a conversation as running.

    Args:
        run_id: The simulation run ID
        index: The 1-based conversation index
        persona: The persona for this conversation
    """
    with _lock:
        if run_id not in _active_runs:
            return

        persona_dict = _persona_to_dict(persona)
        _active_runs[run_id]["active"][index] = {
            "persona_name": persona_dict.get("name", f"Persona {index}"),
            "turn_count": 0,
            "status": "running",
        }


def update_turn_count(run_id: int, index: int, turn_count: int) -> None:
    """
    Update the turn count for a running conversation.

    Args:
        run_id: The simulation run ID
        index: The 1-based conversation index
        turn_count: Current number of turns completed
    """
    with _lock:
        if run_id not in _active_runs:
            return
        if index not in _active_runs[run_id]["active"]:
            return

        _active_runs[run_id]["active"][index]["turn_count"] = turn_count


def complete_conversation(run_id: int, index: int) -> None:
    """
    Mark a conversation as completed.

    Moves the conversation from active to completed_indices.

    Args:
        run_id: The simulation run ID
        index: The 1-based conversation index
    """
    with _lock:
        if run_id not in _active_runs:
            return

        # Remove from active
        if index in _active_runs[run_id]["active"]:
            del _active_runs[run_id]["active"][index]

        # Add to completed
        _active_runs[run_id]["completed_indices"].add(index)


def get_run_progress(run_id: int) -> dict[str, Any] | None:
    """
    Get all conversation statuses for a run.

    Args:
        run_id: The simulation run ID

    Returns:
        Dict with personas, active conversations, and completed indices,
        or None if run is not tracked.
    """
    with _lock:
        if run_id not in _active_runs:
            return None

        run = _active_runs[run_id]
        return {
            "phase": run.get("phase", "running"),
            "target_count": run.get("target_count", len(run["personas"])),
            "personas": run["personas"],
            "active": dict(run["active"]),
            "completed_indices": set(run["completed_indices"]),
        }


def cleanup_run(run_id: int) -> None:
    """
    Remove a completed run from tracking.

    Called when a run finishes (success or failure).

    Args:
        run_id: The simulation run ID
    """
    with _lock:
        if run_id in _active_runs:
            del _active_runs[run_id]


def request_cancel(run_id: int) -> None:
    """
    Request cancellation of a run.

    Sets the cancellation flag and removes from active tracking.
    Background threads should check is_cancelled() cooperatively.

    Args:
        run_id: The simulation run ID
    """
    with _lock:
        _cancelled_runs.add(run_id)
        if run_id in _active_runs:
            del _active_runs[run_id]


def is_cancelled(run_id: int) -> bool:
    """
    Check if a run has been cancelled.

    Called by background threads to cooperatively stop processing.

    Args:
        run_id: The simulation run ID

    Returns:
        True if the run has been cancelled.
    """
    with _lock:
        return run_id in _cancelled_runs


def clear_cancelled(run_id: int) -> None:
    """
    Remove a run from the cancelled set.

    Called after a thread has acknowledged the cancellation.

    Args:
        run_id: The simulation run ID
    """
    with _lock:
        _cancelled_runs.discard(run_id)


def is_run_active(run_id: int) -> bool:
    """
    Check if a run is actively being tracked in memory.

    If a run shows as 'running' or 'judging' in the DB but is NOT active
    in memory, the background thread has died and the run is orphaned.

    Args:
        run_id: The simulation run ID

    Returns:
        True if the run is being tracked (thread is alive).
    """
    with _lock:
        return run_id in _active_runs


def start_judging_conversation(conversation_id: int) -> None:
    """Mark a conversation as currently being judged."""
    with _lock:
        _judging_conversations[conversation_id] = "judging"


def complete_judging_conversation(conversation_id: int, error: bool = False) -> None:
    """Mark a conversation judging as finished."""
    with _lock:
        _judging_conversations[conversation_id] = "error" if error else "done"


def get_judging_status(conversation_ids: list[int]) -> dict[int, str]:
    """Get judging status for a list of conversation IDs.

    Returns dict of conversation_id -> status for any that are tracked.
    Conversations not in the tracker are omitted.
    """
    with _lock:
        return {
            cid: _judging_conversations[cid]
            for cid in conversation_ids
            if cid in _judging_conversations
        }


def clear_judging_conversation(conversation_id: int) -> None:
    """Remove a conversation from the judging tracker."""
    with _lock:
        _judging_conversations.pop(conversation_id, None)


def _persona_to_dict(persona: Any) -> dict[str, Any]:
    """Convert a persona object to a dict for storage."""
    if isinstance(persona, dict):
        return persona

    # Handle persona objects with attributes
    return {
        "name": getattr(persona, "name", "Unknown"),
        "language": getattr(persona.primary_language, "value", "en") if hasattr(persona, "primary_language") else "en",
        "legal_issue": getattr(persona.legal_issue, "value", "unknown") if hasattr(persona, "legal_issue") else "unknown",
    }
