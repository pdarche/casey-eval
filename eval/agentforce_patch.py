"""
Monkey-patch the agentforce SDK to handle missing fields in API responses.

The SDK does hard key lookups (msg["planId"], msg["feedbackId"], etc.) when
parsing messages. The AgentForce API sometimes omits these fields (e.g. at
conversation end), causing KeyError. This patch replaces the parsing with
resilient .get() calls.

Call apply() once at startup before any agentforce usage.
"""

_patched = False


def apply():
    """Apply resilient message parsing to the agentforce SDK."""
    import sys
    global _patched
    if _patched:
        return
    _patched = True
    print("[agentforce_patch] Applying SDK patch", file=sys.stderr, flush=True)

    try:
        import requests
        import uuid
        import agentforce.rest.send_message as send_msg_mod
        import agentforce.rest.start_session as start_sess_mod
        from agentforce.constant.constants import (
            CONTINUE_SESSION_URL, START_SESSION_URL,
            VARIABLES_TEMPLATE, TIMEZONE, FEATURE_SUPPORT,
        )
        from agentforce.data.message import SendMessageResponse, MessageResponse, Links, Link
        from agentforce.data.session import SessionResponse, Links as SessionLinks, Message, Link as SessionLink
    except ImportError:
        return

    def _parse_message_response(msg):
        return MessageResponse(
            type=msg.get("type", ""),
            id=msg.get("id", ""),
            feedbackId=msg.get("feedbackId", ""),
            planId=msg.get("planId", ""),
            isContentSafe=msg.get("isContentSafe", True),
            message=msg.get("message", ""),
            result=msg.get("result", []),
            citedReferences=msg.get("citedReferences", []),
        )

    def _parse_links(data):
        links_data = data.get("_links", {})
        return Links(
            self=Link(href=links_data.get("self", "")),
            messages=Link(href=links_data.get("messages", {}).get("href", "")),
            messagesStream=Link(href=links_data.get("messagesStream", {}).get("href", "")),
            session=Link(href=links_data.get("session", {}).get("href", "")),
            end=Link(href=links_data.get("end", {}).get("href", "")),
        )

    def patched_send_message(instance_url, access_token, session_id, message):
        url = CONTINUE_SESSION_URL.replace("{session-id}", session_id)
        payload = {"message": message, "variables": VARIABLES_TEMPLATE}
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        return SendMessageResponse(
            messages=[_parse_message_response(m) for m in data.get("messages", [])],
            _links=_parse_links(data),
        )

    def _parse_session_links(data):
        links_data = data.get("_links", {})
        return SessionLinks(
            self=SessionLink(href=links_data.get("self", "")),
            messages=SessionLink(href=links_data.get("messages", {}).get("href", "")),
            messagesStream=SessionLink(href=links_data.get("messagesStream", {}).get("href", "")),
            session=SessionLink(href=links_data.get("session", {}).get("href", "")),
            end=SessionLink(href=links_data.get("end", {}).get("href", "")),
        )

    def patched_start_session(instance_url, access_token, agent_id):
        url = START_SESSION_URL.format(agentId=agent_id)
        payload = {
            "externalSessionKey": str(uuid.uuid4()),
            "instanceConfig": {"endpoint": instance_url},
            "tz": TIMEZONE,
            "variables": VARIABLES_TEMPLATE,
            "featureSupport": FEATURE_SUPPORT,
            "streamingCapabilities": {"chunkTypes": ["Text"]},
            "bypassUser": "true",
        }
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        messages = [
            Message(
                type=m.get("type", ""),
                id=m.get("id", ""),
                feedbackId=m.get("feedbackId", ""),
                planId=m.get("planId", ""),
                isContentSafe=m.get("isContentSafe", True),
                message=m.get("message", ""),
                result=m.get("result", []),
                citedReferences=m.get("citedReferences", []),
            )
            for m in data.get("messages", [])
        ]

        return SessionResponse(
            sessionId=data["sessionId"],
            _links=_parse_session_links(data),
            messages=messages,
        )

    send_msg_mod.send_message = patched_send_message
    start_sess_mod.start_session = patched_start_session

    # Also patch the references in agentforce.agents, since it does
    # `from .rest.send_message import send_message` which binds a local copy
    try:
        import agentforce.agents as agents_mod
        agents_mod.send_message = patched_send_message
        agents_mod.start_session = patched_start_session
    except ImportError:
        pass
