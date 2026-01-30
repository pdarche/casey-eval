"""
Synthetic client that role-plays personas for conversation simulation.

The synthetic client uses an LLM to generate realistic responses
based on a persona's characteristics and the conversation context.
"""

from string import Template
from typing import Optional
from dataclasses import dataclass, field

from eval.personas.models import Persona


# System prompt for the synthetic client LLM
SYNTHETIC_CLIENT_SYSTEM_PROMPT = Template("""You are role-playing as a person seeking free legal aid from Open Door Legal in San Francisco.

${persona_context}

INSTRUCTIONS:
- Stay in character throughout the conversation
- Answer questions honestly based on your persona
- Communicate at an 8th grade reading/writing level
- Only respond to what the agent asks - don't volunteer extra information unprompted
- If you don't understand a question, ask for clarification
- Express appropriate emotions (stress, confusion, relief) naturally
- When the agent says the intake is complete or schedules a follow-up, respond with "INTAKE_COMPLETE"

IMPORTANT:
- You are the CLIENT, not the legal aid worker
- Wait for questions before providing information
- Keep responses concise but natural
- If speaking in a non-English language, respond in that language

Previous messages in this conversation are provided for context. Generate your next response as the client.""")


SYNTHETIC_CLIENT_USER_PROMPT = Template("""CONVERSATION SO FAR:
${conversation_history}

CASEY (Agent) just said:
${agent_message}

Generate your response as ${persona_name}. Remember to stay in character and respond naturally.
If the intake appears complete (agent summarized your info and scheduled follow-up), respond with just "INTAKE_COMPLETE".

Your response:""")


@dataclass
class SyntheticClient:
    """
    Generates realistic client responses based on a persona.

    Uses an LLM to role-play as the persona, maintaining consistency
    throughout the conversation.
    """

    persona: Persona
    llm_client: any  # OpenAI client
    model: str = "gpt-4.1-mini"
    temperature: float = 0.7
    conversation_history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        """Initialize the system prompt based on persona."""
        self.system_prompt = SYNTHETIC_CLIENT_SYSTEM_PROMPT.substitute(
            persona_context=self.persona.get_system_prompt_context()
        )

    def _format_conversation_history(self) -> str:
        """Format conversation history for the prompt."""
        if not self.conversation_history:
            return "(This is the start of the conversation)"

        lines = []
        for msg in self.conversation_history:
            role = "CASEY" if msg["role"] == "assistant" else "YOU"
            lines.append(f"{role}: {msg['content']}")

        return "\n\n".join(lines)

    def generate_response(self, agent_message: str) -> str:
        """
        Generate a client response to the agent's message.

        Args:
            agent_message: The message from Casey to respond to

        Returns:
            The synthetic client's response
        """
        # Build the user prompt
        user_prompt = SYNTHETIC_CLIENT_USER_PROMPT.substitute(
            conversation_history=self._format_conversation_history(),
            agent_message=agent_message,
            persona_name=self.persona.name,
        )

        # Call LLM
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=500,
        )

        client_response = response.choices[0].message.content.strip()

        # Update conversation history
        self.conversation_history.append({"role": "assistant", "content": agent_message})
        self.conversation_history.append({"role": "user", "content": client_response})

        return client_response

    def reset(self):
        """Reset conversation history for a new conversation."""
        self.conversation_history = []

    def get_transcript(self) -> str:
        """Get the full conversation transcript."""
        lines = []
        for msg in self.conversation_history:
            role = "Casey" if msg["role"] == "assistant" else "Client"
            lines.append(f"{role}: {msg['content']}")
        return "\n\n".join(lines)
