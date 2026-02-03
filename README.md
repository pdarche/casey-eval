# Casey Evaluation Suite

Offline evaluation framework for Casey, the Open Door Legal Salesforce intake agent.

## Quick Start

```bash
# Install dependencies
uv sync

# Set environment variables for Salesforce Agentforce API
export CASEY_API_URL="https://your-org.my.salesforce.com"
export CASEY_API_KEY="your-salesforce-access-token"
export AGENTFORCE_AGENT_ID="your-18-char-agent-id"
export OPENAI_API_KEY="your-openai-key"

# Run evaluation
uv run casey-eval run-eval --conversations 10

# Or use python -m
uv run python -m eval.cli run-eval --conversations 10
```

## Salesforce Agentforce Setup

1. Create a Connected App in Salesforce Setup with Client Credentials Flow
2. Get an access token using the OAuth endpoint
3. Find your Agent ID from the Agent Overview page URL

See [docs/AGENT_EVALUATION_FRAMEWORK.md](docs/AGENT_EVALUATION_FRAMEWORK.md) for detailed setup instructions.

## Commands

```bash
# Run full evaluation suite
uv run casey-eval run-eval --conversations 10 --output results.json

# Test single persona
uv run casey-eval run-single --persona "Maria Santos" --preview

# List available personas
uv run casey-eval list-personas

# List behavioral rules
uv run casey-eval list-rules

# Generate report from results
uv run casey-eval report results.json
```

## Usage Workflow

### 0. Update the Prompt in Salesforce
Based on feedback from prior evaluation runs, revise the Casey agent prompt in Salesforce Agentforce.

### 1. Create a New Prompt Version
Add the updated Salesforce prompt to the evaluation suite by creating a new prompt version:
```bash
uv run casey-eval add-prompt --version "v2.1" --file prompt.txt
```

### 2. Create a New Evaluation Run
Run an evaluation with your desired parameters:
```bash
uv run casey-eval run-eval --conversations 10 --prompt-version "v2.1" --output results.json
```

### 3. Review and Provide Feedback
Review the evaluation results and provide feedback to inform the next prompt iteration:
```bash
uv run casey-eval report results.json
```

## Documentation

See [docs/AGENT_EVALUATION_FRAMEWORK.md](docs/AGENT_EVALUATION_FRAMEWORK.md) for full documentation.
