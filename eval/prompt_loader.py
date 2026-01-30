"""
Prompt loader utility for accessing Casey's prompt programmatically.

This module provides utilities to:
1. Load the full prompt from the odl_chat repository
2. Extract behavioral instructions from the prompt
3. Parse the intake script JSON for evaluation
"""

import json
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


# Path to sibling odl_chat repository
ODL_CHAT_PATH = Path(__file__).parent.parent.parent / "odl_chat"


@dataclass
class ExtractedInstruction:
    """An instruction extracted from the prompt."""
    order: int
    title: str
    content: str
    instruction_type: str


@dataclass
class ExtractedQuestion:
    """A question extracted from the intake script."""
    question_id: str
    section: str
    order: int
    text: str
    required: bool
    response_format: dict
    sync_to_salesforce: bool
    salesforce_api_name: Optional[str]
    tools_to_call: list[str]
    instructions: Optional[str]
    skip_if_existing_user: bool = False


@dataclass
class ParsedScript:
    """Parsed intake script with all components."""
    name: str
    description: str
    supported_languages: list[str]
    instructions: list[ExtractedInstruction]
    questions: list[ExtractedQuestion]
    sections: list[dict]


def get_odl_chat_path() -> Path:
    """Get the path to the odl_chat repository."""
    if ODL_CHAT_PATH.exists():
        return ODL_CHAT_PATH
    raise FileNotFoundError(f"odl_chat repository not found at {ODL_CHAT_PATH}")


def load_intake_script_json() -> dict:
    """
    Load the raw intake script JSON from odl_chat.

    Returns:
        dict: The parsed JSON content of odl_intake_script.json
    """
    script_path = get_odl_chat_path() / "data" / "odl_intake_script.json"
    if not script_path.exists():
        raise FileNotFoundError(f"Intake script not found at {script_path}")

    with open(script_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_intake_script() -> ParsedScript:
    """
    Parse the intake script JSON into structured components.

    Returns:
        ParsedScript with instructions, questions, and sections
    """
    raw = load_intake_script_json()
    script = raw.get("script", {})

    # Parse metadata
    metadata = script.get("metadata", {})
    supported_languages = metadata.get("supported_languages", [])

    # Parse instructions
    instructions = []
    for inst in script.get("instructions", []):
        instructions.append(ExtractedInstruction(
            order=inst.get("order", 0),
            title=inst.get("title", ""),
            content=inst.get("content", ""),
            instruction_type=inst.get("instruction_type", "general"),
        ))

    # Parse questions from all sections
    questions = []
    sections = []

    for section in script.get("sections", []):
        section_name = section.get("name", "")
        sections.append({
            "name": section_name,
            "order": section.get("order", 0),
            "description": section.get("description", ""),
            "save_responses_after": section.get("save_responses_after", False),
        })

        for q in section.get("questions", []):
            questions.append(ExtractedQuestion(
                question_id=q.get("question_id", ""),
                section=section_name,
                order=q.get("order", 0),
                text=q.get("text", ""),
                required=q.get("required", False),
                response_format=q.get("response_format", {}),
                sync_to_salesforce=q.get("sync_to_salesforce", False),
                salesforce_api_name=q.get("salesforce_api_name"),
                tools_to_call=q.get("tools_to_call", []),
                instructions=q.get("instructions"),
                skip_if_existing_user=q.get("skip_if_existing_user", False),
            ))

    return ParsedScript(
        name=script.get("name", ""),
        description=script.get("description", ""),
        supported_languages=supported_languages,
        instructions=sorted(instructions, key=lambda x: x.order),
        questions=sorted(questions, key=lambda x: (x.section, x.order)),
        sections=sorted(sections, key=lambda x: x["order"]),
    )


def load_full_prompt() -> str:
    """
    Load the full Casey prompt from the odl_chat module.

    This imports from the odl_chat repository and generates the template.

    Returns:
        str: The full system prompt
    """
    # Add odl_chat to path if not present
    odl_chat_path = str(get_odl_chat_path())
    if odl_chat_path not in sys.path:
        sys.path.insert(0, odl_chat_path)

    try:
        from assistant.template_generator import TemplateGenerator

        with TemplateGenerator() as generator:
            return generator.generate_template()
    except ImportError as e:
        raise ImportError(f"Could not import from odl_chat: {e}")


def load_abbreviated_prompt() -> str:
    """
    Load the abbreviated (follow-up) prompt from odl_chat.

    Returns:
        str: The abbreviated system prompt for follow-up conversations
    """
    odl_chat_path = str(get_odl_chat_path())
    if odl_chat_path not in sys.path:
        sys.path.insert(0, odl_chat_path)

    try:
        from assistant.template_generator import TemplateGenerator

        with TemplateGenerator() as generator:
            return generator.generate_abbreviated_template()
    except ImportError as e:
        raise ImportError(f"Could not import from odl_chat: {e}")


def extract_testable_behaviors_from_script() -> list[dict]:
    """
    Extract testable behavioral expectations from the intake script.

    Parses the instructions and question-level instructions to generate
    a list of testable behaviors for evaluation.

    Returns:
        List of dicts describing testable behaviors
    """
    script = parse_intake_script()
    behaviors = []

    # Extract from main instructions
    for inst in script.instructions:
        behaviors.append({
            "source": "instruction",
            "source_id": f"instruction_{inst.order}",
            "title": inst.title,
            "description": inst.content,
            "type": inst.instruction_type,
            "testable": True,
        })

    # Extract from question-level instructions
    for q in script.questions:
        if q.instructions:
            behaviors.append({
                "source": "question",
                "source_id": q.question_id,
                "title": f"Question {q.question_id} handling",
                "description": q.instructions,
                "type": "question_specific",
                "section": q.section,
                "testable": True,
            })

        # Add format validation requirements
        if q.response_format:
            behaviors.append({
                "source": "format",
                "source_id": f"{q.question_id}_format",
                "title": f"Question {q.question_id} format validation",
                "description": f"Response must be type: {q.response_format.get('type', 'unknown')}",
                "type": "data_validation",
                "section": q.section,
                "response_format": q.response_format,
                "testable": True,
            })

    return behaviors


def get_questions_by_section() -> dict[str, list[ExtractedQuestion]]:
    """
    Get questions organized by section.

    Returns:
        Dict mapping section names to lists of questions
    """
    script = parse_intake_script()
    by_section = {}

    for q in script.questions:
        if q.section not in by_section:
            by_section[q.section] = []
        by_section[q.section].append(q)

    return by_section


def get_required_questions() -> list[ExtractedQuestion]:
    """
    Get all required questions.

    Returns:
        List of required questions
    """
    script = parse_intake_script()
    return [q for q in script.questions if q.required]


def get_salesforce_mapped_questions() -> list[ExtractedQuestion]:
    """
    Get questions that sync to Salesforce.

    Returns:
        List of questions with Salesforce mapping
    """
    script = parse_intake_script()
    return [q for q in script.questions if q.sync_to_salesforce]


# Quick test when run directly
if __name__ == "__main__":
    print("Testing prompt loader...")

    try:
        script = parse_intake_script()
        print(f"\nScript: {script.name}")
        print(f"Languages: {script.supported_languages}")
        print(f"Instructions: {len(script.instructions)}")
        print(f"Questions: {len(script.questions)}")
        print(f"Sections: {len(script.sections)}")

        print("\n--- Instructions ---")
        for inst in script.instructions[:3]:
            print(f"  {inst.order}. {inst.title} ({inst.instruction_type})")

        print("\n--- Required Questions ---")
        required = get_required_questions()
        print(f"  {len(required)} required questions")

        behaviors = extract_testable_behaviors_from_script()
        print(f"\n--- Testable Behaviors ---")
        print(f"  {len(behaviors)} testable behaviors extracted")

    except Exception as e:
        print(f"Error: {e}")
