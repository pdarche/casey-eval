#!/usr/bin/env python3
"""
Flask webapp for viewing Casey evaluation results.

Usage:
    uv run python webapp/app.py
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from webapp.utils import (
    list_versions,
    get_version,
    get_conversation,
    get_conversation_by_file,
    get_transcripts_dir,
)

app = Flask(__name__)


@app.route("/")
def index():
    """Dashboard showing all versions."""
    versions = list_versions()
    return render_template("index.html", versions=versions)


@app.route("/version/<version_id>")
def version_detail(version_id: str):
    """Detail page for a specific version."""
    version = get_version(version_id)
    if not version:
        return "Version not found", 404
    return render_template("version.html", version=version)


@app.route("/conversation/<session_id>")
def conversation_detail(session_id: str):
    """Conversation inspector."""
    conversation = get_conversation(session_id)
    if not conversation:
        return "Conversation not found", 404
    return render_template("conversation.html", conversation=conversation)


@app.route("/conversation-by-file")
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
        safety_eval = SafetyEvaluator(llm_client=llm_client, model="gpt-4o")
        quality_eval = QualityEvaluator(llm_client=llm_client, model="gpt-4o")
        completeness_eval = CompletenessEvaluator(llm_client=llm_client, model="gpt-4o")

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
