#!/usr/bin/env python3
"""
Test Agentforce API using the salesforce-agentforce SDK
"""

from agent_sdk import Agentforce

# Configuration - using original working credentials
# Note: SDK expects domain only, not full URL
SALESFORCE_ORG = "opendoorlegal--sfaccel.sandbox.my.salesforce.com"
CLIENT_ID = "REDACTED_CLIENT_ID"
CLIENT_SECRET = "REDACTED_CLIENT_SECRET"
AGENT_ID = "0XxO800000037BRKAY"

print("=" * 60)
print("Testing Agentforce SDK")
print("=" * 60)
print(f"Salesforce Org: {SALESFORCE_ORG}")
print(f"Agent ID: {AGENT_ID}")
print()

# Initialize client
client = Agentforce()

# Step 1: Authenticate
print("Step 1: Authenticating...")
try:
    client.authenticate(
        salesforce_org=SALESFORCE_ORG,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )
    print(f"  ✓ Authenticated successfully")
    print(f"  Instance URL: {client.instance_url}")
except Exception as e:
    print(f"  ✗ Authentication failed: {e}")
    exit(1)

# Step 2: Start session
print("\nStep 2: Starting session...")
try:
    session = client.start_session(agent_id=AGENT_ID)
    print(f"  ✓ Session started")
    print(f"  Session ID: {session.sessionId}")
except Exception as e:
    print(f"  ✗ Failed to start session: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 3: Send a message
print("\nStep 3: Sending test message...")
try:
    client.add_message_text("Hello, I need help with a housing issue.")
    response = client.send_message(session_id=session.sessionId)
    print(f"  ✓ Message sent")
    print(f"  Response: {response}")
except Exception as e:
    print(f"  ✗ Failed to send message: {e}")
    import traceback
    traceback.print_exc()

# Step 4: End session
print("\nStep 4: Ending session...")
try:
    end_response = client.end_session(session_id=session.sessionId)
    print(f"  ✓ Session ended")
except Exception as e:
    print(f"  ✗ Failed to end session: {e}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
