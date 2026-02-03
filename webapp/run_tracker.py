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


def register_run(run_id: int, personas: list[Any]) -> None:
    """
    Register a new run with its persona list.

    Called when a run starts, before any conversations begin.

    Args:
        run_id: The simulation run ID
        personas: List of persona objects/dicts for this run
    """
    with _lock:
        _active_runs[run_id] = {
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
