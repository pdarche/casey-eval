"""
Conversation runner that orchestrates evaluation conversations.

Manages the interaction between the synthetic client and Casey
via Salesforce Agentforce API, collecting the full conversation for evaluation.

Agentforce API Reference:
https://developer.salesforce.com/docs/ai/agentforce/guide/agent-api.html
"""

import httpx
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import datetime

from eval.personas.models import Persona
from eval.simulation.client import SyntheticClient
from eval.judges.base import ConversationContext, ConversationTurn


@dataclass
class ConversationResult:
    """Result of a completed conversation."""

    thread_id: str  # Agentforce session_id
    persona: Persona
    turns: list[ConversationTurn]
    completed: bool
    completion_reason: str  # "intake_complete", "max_turns", "error", "timeout"
    duration_seconds: float
    turn_count: int
    error: Optional[str] = None
    saved_data: Optional[dict] = None
    salesforce_data: Optional[dict] = None
    metadata: dict = field(default_factory=dict)

    def to_context(self) -> ConversationContext:
        """Convert to ConversationContext for evaluation."""
        return ConversationContext(
            turns=self.turns,
            persona=self.persona.to_dict(),
            saved_data=self.saved_data,
            salesforce_data=self.salesforce_data,
        )

    def get_transcript(self) -> str:
        """Get formatted transcript."""
        lines = []
        for turn in self.turns:
            role = "Client" if turn.role == "user" else "Casey"
            lines.append(f"{role}: {turn.content}")
        return "\n\n".join(lines)


@dataclass
class AgentforceConfig:
    """Configuration for Salesforce Agentforce Agent API.

    Requires:
    - my_domain_url: Your Salesforce My Domain URL (e.g., https://your-org.my.salesforce.com)
    - agent_id: 18-character Agent ID from the Agent Overview page
    - access_token: OAuth access token from connected app (client credentials flow)

    See: https://developer.salesforce.com/docs/ai/agentforce/guide/agent-api-get-started.html
    """

    my_domain_url: str
    agent_id: str
    access_token: str
    timeout: float = 120.0  # Agentforce has 120-second timeout
    api_version: str = "v61.0"
    bypass_user: bool = True  # Required for client credentials flow
    use_streaming: bool = False  # Use synchronous endpoint by default

    def get_headers(self, streaming: bool = False) -> dict:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }
        if streaming:
            headers["Accept"] = "text/event-stream"
        return headers

    @property
    def sessions_url(self) -> str:
        """URL for creating new sessions."""
        return f"{self.my_domain_url}/einstein/ai-agent/v1/agents/{self.agent_id}/sessions"

    def messages_url(self, session_id: str, sync: bool = True) -> str:
        """URL for sending messages to a session."""
        endpoint = "messages" if sync else "messagesStream"
        sync_param = "?sync=true" if sync else ""
        return f"{self.my_domain_url}/einstein/ai-agent/v1/sessions/{session_id}/{endpoint}{sync_param}"

    def end_session_url(self, session_id: str) -> str:
        """URL for ending a session."""
        return f"{self.my_domain_url}/einstein/ai-agent/v1/sessions/{session_id}"


# Keep legacy config for backward compatibility
@dataclass
class CaseyAPIConfig:
    """Legacy configuration - use AgentforceConfig for Salesforce Agentforce API."""

    base_url: str
    api_key: Optional[str] = None
    timeout: float = 60.0
    headers: dict = field(default_factory=dict)
    start_conversation_path: str = "/api/conversations"
    send_message_path: str = "/api/conversations/{thread_id}/messages"
    get_conversation_path: str = "/api/conversations/{thread_id}"

    def get_headers(self) -> dict:
        """Get headers for API requests."""
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class AgentforceConversationRunner:
    """
    Runs evaluation conversations using the Salesforce Agentforce Agent API.

    Implements the full session lifecycle:
    1. Start session with agent
    2. Send/receive messages with sequence tracking
    3. End session when complete

    See: https://developer.salesforce.com/docs/ai/agentforce/guide/agent-api-examples.html
    """

    def __init__(
        self,
        config: AgentforceConfig,
        llm_client,
        synthetic_model: str = "gpt-4.1-mini",
    ):
        """
        Initialize the Agentforce conversation runner.

        Args:
            config: Agentforce API configuration
            llm_client: OpenAI client for synthetic client
            synthetic_model: Model to use for synthetic client responses
        """
        self.config = config
        self.llm_client = llm_client
        self.synthetic_model = synthetic_model
        self.http_client = httpx.Client(timeout=config.timeout)

    def _start_session(self) -> dict:
        """Start a new session with the Agentforce agent.

        Returns dict with:
        - sessionId: The session identifier for subsequent calls
        - messages: Initial agent messages (greeting)
        - _links: URLs for messages, end session, etc.
        """
        external_session_key = str(uuid.uuid4())

        payload = {
            "externalSessionKey": external_session_key,
            "instanceConfig": {
                "endpoint": self.config.my_domain_url
            },
            "streamingCapabilities": {
                "chunkTypes": ["Text"]
            },
            "bypassUser": self.config.bypass_user
        }

        response = self.http_client.post(
            self.config.sessions_url,
            headers=self.config.get_headers(),
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def _send_message(self, session_id: str, message: str, sequence_id: int) -> dict:
        """Send a message to the agent and get response.

        Args:
            session_id: The session ID from _start_session
            message: The user message text
            sequence_id: Incrementing sequence number for this session

        Returns dict with:
        - messages: List of agent response messages
        """
        payload = {
            "message": {
                "sequenceId": sequence_id,
                "type": "Text",
                "text": message
            },
            "variables": []
        }

        url = self.config.messages_url(session_id, sync=not self.config.use_streaming)

        response = self.http_client.post(
            url,
            headers=self.config.get_headers(streaming=self.config.use_streaming),
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def _end_session(self, session_id: str) -> bool:
        """End the agent session.

        Args:
            session_id: The session ID to end

        Returns:
            True if session ended successfully
        """
        try:
            response = self.http_client.delete(
                self.config.end_session_url(session_id),
                headers=self.config.get_headers(),
            )
            return response.status_code in (200, 204)
        except Exception:
            return False

    def _extract_agent_message(self, response: dict) -> str:
        """Extract agent message text from API response.

        Handles different response formats from Agentforce API.
        """
        messages = response.get("messages", [])
        if not messages:
            return ""

        # Combine all text messages from the response
        text_parts = []
        for msg in messages:
            if msg.get("type") == "Text":
                text_parts.append(msg.get("text", ""))
            elif msg.get("type") == "Inform":
                # Inform messages use 'message' field, not 'text'
                text_parts.append(msg.get("message", msg.get("text", "")))

        return " ".join(text_parts).strip()

    def run(
        self,
        persona: Persona,
        max_turns: int = 100,
        turn_delay: float = 0.5,
        on_turn: Optional[Callable[[int, str, str], None]] = None,
    ) -> ConversationResult:
        """
        Run a full conversation with the given persona.

        Args:
            persona: The persona for the synthetic client
            max_turns: Maximum number of turns before stopping
            turn_delay: Delay between turns (seconds)
            on_turn: Optional callback called after each turn

        Returns:
            ConversationResult with full conversation and metadata
        """
        start_time = time.time()
        turns: list[ConversationTurn] = []
        session_id = ""
        completion_reason = "max_turns"
        error = None
        sequence_id = 0

        # Create synthetic client
        synthetic_client = SyntheticClient(
            persona=persona,
            llm_client=self.llm_client,
            model=self.synthetic_model,
        )

        try:
            # Start session with Agentforce
            session_response = self._start_session()
            session_id = session_response.get("sessionId", "")

            # Get initial greeting from agent
            agent_message = self._extract_agent_message(session_response)

            if agent_message:
                turns.append(ConversationTurn(
                    role="assistant",
                    content=agent_message,
                    timestamp=datetime.now().isoformat(),
                ))

            turn_count = 0

            while turn_count < max_turns:
                turn_count += 1
                sequence_id += 1

                # Generate client response
                client_response = synthetic_client.generate_response(agent_message)

                turns.append(ConversationTurn(
                    role="user",
                    content=client_response,
                    timestamp=datetime.now().isoformat(),
                ))

                # Check for completion signal from synthetic client
                if "INTAKE_COMPLETE" in client_response.upper():
                    completion_reason = "intake_complete"
                    break

                # Small delay to avoid rate limiting
                if turn_delay > 0:
                    time.sleep(turn_delay)

                # Send to agent and get response
                response = self._send_message(session_id, client_response, sequence_id)
                agent_message = self._extract_agent_message(response)

                turns.append(ConversationTurn(
                    role="assistant",
                    content=agent_message,
                    timestamp=datetime.now().isoformat(),
                ))

                # Callback if provided
                if on_turn:
                    on_turn(turn_count, client_response, agent_message)

                # Check if agent indicates completion
                completion_indicators = [
                    "intake is complete",
                    "form is complete",
                    "thank you for completing",
                    "we'll follow up",
                    "within 2 business days",
                ]
                if any(ind in agent_message.lower() for ind in completion_indicators):
                    # Give client one more turn to acknowledge
                    client_response = synthetic_client.generate_response(agent_message)
                    turns.append(ConversationTurn(
                        role="user",
                        content=client_response,
                        timestamp=datetime.now().isoformat(),
                    ))
                    completion_reason = "intake_complete"
                    break

            # End the session
            self._end_session(session_id)

        except httpx.TimeoutException:
            completion_reason = "timeout"
            error = "HTTP request timed out (Agentforce has 120s limit)"
        except httpx.HTTPStatusError as e:
            completion_reason = "error"
            error = f"HTTP error: {e.response.status_code} - {e.response.text[:200]}"
        except Exception as e:
            completion_reason = "error"
            error = str(e)

        duration = time.time() - start_time

        return ConversationResult(
            thread_id=session_id,
            persona=persona,
            turns=turns,
            completed=(completion_reason == "intake_complete"),
            completion_reason=completion_reason,
            duration_seconds=duration,
            turn_count=len([t for t in turns if t.role == "user"]),
            error=error,
            metadata={
                "synthetic_model": self.synthetic_model,
                "max_turns": max_turns,
                "api_type": "agentforce",
            },
        )

    def run_batch(
        self,
        personas: list[Persona],
        max_turns: int = 100,
        turn_delay: float = 0.5,
        on_conversation_complete: Optional[Callable[[int, ConversationResult], None]] = None,
    ) -> list[ConversationResult]:
        """Run conversations for multiple personas."""
        results = []

        for i, persona in enumerate(personas):
            result = self.run(
                persona=persona,
                max_turns=max_turns,
                turn_delay=turn_delay,
            )
            results.append(result)

            if on_conversation_complete:
                on_conversation_complete(i, result)

        return results

    def close(self):
        """Close HTTP client."""
        self.http_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ConversationRunner:
    """
    Legacy conversation runner for generic HTTP APIs.

    For Salesforce Agentforce, use AgentforceConversationRunner instead.
    """

    def __init__(
        self,
        api_config: CaseyAPIConfig,
        llm_client,
        synthetic_model: str = "gpt-4.1-mini",
    ):
        """
        Initialize the conversation runner.

        Args:
            api_config: Configuration for Casey's HTTP API
            llm_client: OpenAI client for synthetic client
            synthetic_model: Model to use for synthetic client responses
        """
        self.api_config = api_config
        self.llm_client = llm_client
        self.synthetic_model = synthetic_model
        self.http_client = httpx.Client(timeout=api_config.timeout)

    def _start_conversation(self) -> dict:
        """Start a new conversation with Casey."""
        url = f"{self.api_config.base_url}{self.api_config.start_conversation_path}"

        response = self.http_client.post(
            url,
            headers=self.api_config.get_headers(),
            json={},
        )
        response.raise_for_status()
        return response.json()

    def _send_message(self, thread_id: str, message: str) -> dict:
        """Send a message to Casey and get response."""
        path = self.api_config.send_message_path.format(thread_id=thread_id)
        url = f"{self.api_config.base_url}{path}"

        response = self.http_client.post(
            url,
            headers=self.api_config.get_headers(),
            json={"message": message},
        )
        response.raise_for_status()
        return response.json()

    def _get_conversation_data(self, thread_id: str) -> dict:
        """Get saved conversation data from Casey."""
        path = self.api_config.get_conversation_path.format(thread_id=thread_id)
        url = f"{self.api_config.base_url}{path}"

        response = self.http_client.get(
            url,
            headers=self.api_config.get_headers(),
        )
        response.raise_for_status()
        return response.json()

    def run(
        self,
        persona: Persona,
        max_turns: int = 100,
        turn_delay: float = 0.5,
        on_turn: Optional[Callable[[int, str, str], None]] = None,
    ) -> ConversationResult:
        """
        Run a full conversation with the given persona.

        Args:
            persona: The persona for the synthetic client
            max_turns: Maximum number of turns before stopping
            turn_delay: Delay between turns (seconds)
            on_turn: Optional callback called after each turn (turn_num, client_msg, agent_msg)

        Returns:
            ConversationResult with full conversation and metadata
        """
        start_time = time.time()
        turns: list[ConversationTurn] = []
        thread_id = ""
        completion_reason = "max_turns"
        error = None

        # Create synthetic client
        synthetic_client = SyntheticClient(
            persona=persona,
            llm_client=self.llm_client,
            model=self.synthetic_model,
        )

        try:
            # Start conversation
            init_response = self._start_conversation()
            thread_id = init_response.get("thread_id", init_response.get("id", ""))

            # Get initial message from Casey
            agent_message = init_response.get("message", init_response.get("response", ""))

            if agent_message:
                turns.append(ConversationTurn(
                    role="assistant",
                    content=agent_message,
                    timestamp=datetime.now().isoformat(),
                ))

            turn_count = 0

            while turn_count < max_turns:
                turn_count += 1

                # Generate client response
                client_response = synthetic_client.generate_response(agent_message)

                turns.append(ConversationTurn(
                    role="user",
                    content=client_response,
                    timestamp=datetime.now().isoformat(),
                ))

                # Check for completion signal
                if "INTAKE_COMPLETE" in client_response.upper():
                    completion_reason = "intake_complete"
                    break

                # Small delay to avoid rate limiting
                if turn_delay > 0:
                    time.sleep(turn_delay)

                # Send to Casey and get response
                response = self._send_message(thread_id, client_response)
                agent_message = response.get("message", response.get("response", ""))

                turns.append(ConversationTurn(
                    role="assistant",
                    content=agent_message,
                    timestamp=datetime.now().isoformat(),
                ))

                # Callback if provided
                if on_turn:
                    on_turn(turn_count, client_response, agent_message)

                # Check if Casey indicates completion
                completion_indicators = [
                    "intake is complete",
                    "form is complete",
                    "thank you for completing",
                    "we'll follow up",
                    "within 2 business days",
                ]
                if any(ind in agent_message.lower() for ind in completion_indicators):
                    # Give client one more turn to acknowledge
                    client_response = synthetic_client.generate_response(agent_message)
                    turns.append(ConversationTurn(
                        role="user",
                        content=client_response,
                        timestamp=datetime.now().isoformat(),
                    ))
                    completion_reason = "intake_complete"
                    break

            # Get final conversation data
            saved_data = None
            try:
                conv_data = self._get_conversation_data(thread_id)
                saved_data = conv_data.get("responses", conv_data.get("data", {}))
            except Exception:
                pass  # Data retrieval is optional

        except httpx.TimeoutException:
            completion_reason = "timeout"
            error = "HTTP request timed out"
        except httpx.HTTPStatusError as e:
            completion_reason = "error"
            error = f"HTTP error: {e.response.status_code}"
        except Exception as e:
            completion_reason = "error"
            error = str(e)

        duration = time.time() - start_time

        return ConversationResult(
            thread_id=thread_id,
            persona=persona,
            turns=turns,
            completed=(completion_reason == "intake_complete"),
            completion_reason=completion_reason,
            duration_seconds=duration,
            turn_count=len([t for t in turns if t.role == "user"]),
            error=error,
            saved_data=saved_data,
            metadata={
                "synthetic_model": self.synthetic_model,
                "max_turns": max_turns,
            },
        )

    def run_batch(
        self,
        personas: list[Persona],
        max_turns: int = 100,
        turn_delay: float = 0.5,
        on_conversation_complete: Optional[Callable[[int, ConversationResult], None]] = None,
    ) -> list[ConversationResult]:
        """
        Run conversations for multiple personas.

        Args:
            personas: List of personas to run conversations for
            max_turns: Maximum turns per conversation
            turn_delay: Delay between turns
            on_conversation_complete: Callback after each conversation

        Returns:
            List of ConversationResults
        """
        results = []

        for i, persona in enumerate(personas):
            result = self.run(
                persona=persona,
                max_turns=max_turns,
                turn_delay=turn_delay,
            )
            results.append(result)

            if on_conversation_complete:
                on_conversation_complete(i, result)

        return results

    def close(self):
        """Close HTTP client."""
        self.http_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
