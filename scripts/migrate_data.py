#!/usr/bin/env python3
"""
Migrate existing JSON files to PostgreSQL.

This script reads existing transcript and eval files from the filesystem
and imports them into the PostgreSQL database.

Usage:
    uv run python scripts/migrate_data.py
    uv run python scripts/migrate_data.py --transcripts-dir ./transcripts
    uv run python scripts/migrate_data.py --dry-run
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.database import (
    init_pool,
    close_pool,
    create_prompt_version,
    get_prompt_version,
    create_simulation_run,
    get_simulation_run_by_version,
    create_conversation,
    get_conversation_by_session_id,
    create_judgment,
)


def load_casey_prompt(data_dir: Path) -> dict:
    """Load the Casey prompt YAML file."""
    prompt_path = data_dir / "casey_prompt_v1.yaml"
    if prompt_path.exists():
        # Read as text since we store the full content
        return {
            "version": "v1.0",
            "name": "Casey Intake Agent",
            "content": prompt_path.read_text(),
        }
    return None


def parse_datetime(dt_str: str) -> datetime:
    """Parse datetime string to datetime object."""
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str)
    except ValueError:
        return None


def migrate_prompt(data_dir: Path, dry_run: bool = False) -> int:
    """Migrate the Casey prompt to the database."""
    prompt_data = load_casey_prompt(data_dir)
    if not prompt_data:
        print("No Casey prompt found")
        return None

    # Check if already exists
    existing = get_prompt_version(prompt_data["version"])
    if existing:
        print(f"Prompt version {prompt_data['version']} already exists (id={existing.id})")
        return existing.id

    if dry_run:
        print(f"[DRY RUN] Would create prompt version: {prompt_data['version']}")
        return -1

    prompt_id = create_prompt_version(
        version=prompt_data["version"],
        name=prompt_data["name"],
        content=prompt_data["content"],
        is_active=True,
    )
    print(f"Created prompt version: {prompt_data['version']} (id={prompt_id})")
    return prompt_id


def migrate_transcripts(
    transcripts_dir: Path,
    prompt_version_id: int = None,
    dry_run: bool = False
) -> dict:
    """
    Migrate transcripts and their evals to the database.

    Returns dict of version -> simulation_run_id
    """
    if not transcripts_dir.exists():
        print(f"Transcripts directory not found: {transcripts_dir}")
        return {}

    # Track simulation runs by version
    simulation_runs = {}
    stats = {
        "conversations_created": 0,
        "conversations_skipped": 0,
        "judgments_created": 0,
        "errors": 0,
    }

    # First, create simulation runs for each version
    # We need to group transcripts by version first
    versions = {}

    for transcript_file in transcripts_dir.glob("*.json"):
        if transcript_file.name.startswith("batch_summary"):
            continue

        try:
            data = json.loads(transcript_file.read_text())
            version = data.get("version") or "unversioned"

            if version not in versions:
                versions[version] = {
                    "transcripts": [],
                    "start_time": None,
                }

            versions[version]["transcripts"].append((transcript_file, data))

            # Track earliest start time
            start_time = parse_datetime(data.get("start_time"))
            if start_time:
                if versions[version]["start_time"] is None:
                    versions[version]["start_time"] = start_time
                elif start_time < versions[version]["start_time"]:
                    versions[version]["start_time"] = start_time

        except json.JSONDecodeError as e:
            print(f"Error reading {transcript_file}: {e}")
            stats["errors"] += 1
            continue

    # Create simulation runs
    for version, info in versions.items():
        # Check if simulation run already exists
        existing = get_simulation_run_by_version(version)
        if existing:
            print(f"Simulation run for version {version} already exists (id={existing.id})")
            simulation_runs[version] = existing.id
            continue

        if dry_run:
            print(f"[DRY RUN] Would create simulation run: {version}")
            simulation_runs[version] = -1
            continue

        run_id = create_simulation_run(
            version=version,
            config={
                "migrated_from_filesystem": True,
                "migration_date": datetime.now().isoformat(),
            },
            prompt_version_id=prompt_version_id,
            status="completed",
        )
        print(f"Created simulation run: {version} (id={run_id})")
        simulation_runs[version] = run_id

    # Now migrate individual conversations
    for version, info in versions.items():
        simulation_run_id = simulation_runs.get(version)
        if simulation_run_id == -1:
            simulation_run_id = None  # For dry run

        for transcript_file, data in info["transcripts"]:
            session_id = data.get("session_id", transcript_file.stem)

            # Check if conversation already exists
            existing = get_conversation_by_session_id(session_id)
            if existing:
                print(f"  Skipping existing conversation: {session_id}")
                stats["conversations_skipped"] += 1
                continue

            if dry_run:
                print(f"  [DRY RUN] Would create conversation: {session_id}")
                stats["conversations_created"] += 1
                continue

            try:
                # Create conversation
                conv_id = create_conversation(
                    session_id=session_id,
                    persona=data.get("persona", {}),
                    transcript=data.get("transcript", []),
                    simulation_run_id=simulation_run_id,
                    completion_reason=data.get("completion_reason"),
                    turn_count=data.get("turn_count"),
                    duration_seconds=data.get("duration_seconds"),
                    start_time=parse_datetime(data.get("start_time")),
                    end_time=parse_datetime(data.get("end_time")),
                    metadata={
                        "original_file": str(transcript_file),
                        "error": data.get("error"),
                    },
                )
                print(f"  Created conversation: {session_id} (id={conv_id})")
                stats["conversations_created"] += 1

                # Load and migrate eval results if they exist
                eval_path = transcript_file.parent / "evals" / f"{transcript_file.stem}_eval.json"
                if eval_path.exists():
                    try:
                        eval_data = json.loads(eval_path.read_text())
                        migrate_eval_results(conv_id, eval_data, dry_run)
                        stats["judgments_created"] += 1
                    except Exception as e:
                        print(f"    Error migrating eval for {session_id}: {e}")
                        stats["errors"] += 1

            except Exception as e:
                print(f"  Error creating conversation {session_id}: {e}")
                stats["errors"] += 1

    return stats


def migrate_eval_results(conversation_id: int, eval_data: dict, dry_run: bool = False) -> None:
    """Migrate eval results for a conversation."""
    # Safety judgments
    for result in eval_data.get("safety", []):
        if dry_run:
            print(f"    [DRY RUN] Would create safety judgment: {result.get('judge_id')}")
            continue

        create_judgment(
            conversation_id=conversation_id,
            judge_type="safety",
            judge_id=result.get("judge_id", "unknown"),
            verdict=result.get("verdict"),
            score=result.get("score"),
            reasoning=result.get("reasoning"),
            evidence=result.get("evidence"),
            metadata=result.get("metadata", {}),
        )

    # Quality judgments
    for result in eval_data.get("quality", []):
        if dry_run:
            print(f"    [DRY RUN] Would create quality judgment: {result.get('judge_id')}")
            continue

        create_judgment(
            conversation_id=conversation_id,
            judge_type="quality",
            judge_id=result.get("judge_id", "unknown"),
            verdict=result.get("verdict"),
            score=result.get("score"),
            reasoning=result.get("reasoning"),
            evidence=result.get("evidence"),
            metadata=result.get("metadata", {}),
        )

    # Completeness judgment
    completeness = eval_data.get("completeness")
    if completeness:
        if dry_run:
            print(f"    [DRY RUN] Would create completeness judgment: {completeness.get('judge_id')}")
            return

        create_judgment(
            conversation_id=conversation_id,
            judge_type="completeness",
            judge_id=completeness.get("judge_id", "completeness_intake"),
            verdict=completeness.get("verdict"),
            score=completeness.get("score"),
            reasoning=completeness.get("reasoning"),
            evidence=completeness.get("evidence"),
            metadata=completeness.get("metadata", {}),
        )


def main():
    parser = argparse.ArgumentParser(description="Migrate JSON files to PostgreSQL")
    parser.add_argument(
        "--transcripts-dir", "-t",
        default="transcripts",
        help="Directory containing transcript JSON files"
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="data",
        help="Directory containing data files (prompts, etc.)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes"
    )

    args = parser.parse_args()

    transcripts_dir = Path(args.transcripts_dir)
    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("ODL Data Migration")
    print("=" * 60)
    print(f"Transcripts directory: {transcripts_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Initialize database connection
    if not args.dry_run:
        print("Initializing database connection...")
        init_pool()

    try:
        # Migrate prompt
        print("\n--- Migrating Casey Prompt ---")
        prompt_id = migrate_prompt(data_dir, args.dry_run)

        # Migrate transcripts
        print("\n--- Migrating Transcripts ---")
        stats = migrate_transcripts(transcripts_dir, prompt_id, args.dry_run)

        # Print summary
        print("\n" + "=" * 60)
        print("MIGRATION SUMMARY")
        print("=" * 60)
        print(f"Conversations created: {stats.get('conversations_created', 0)}")
        print(f"Conversations skipped: {stats.get('conversations_skipped', 0)}")
        print(f"Judgments created: {stats.get('judgments_created', 0)}")
        print(f"Errors: {stats.get('errors', 0)}")

    finally:
        if not args.dry_run:
            close_pool()


if __name__ == "__main__":
    main()
