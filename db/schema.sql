-- ODL Salesforce Agent Evaluation Database Schema
-- This schema supports prompt versioning, simulation runs, conversations, and judgments

-- Prompt versions (Casey's intake script versions)
CREATE TABLE IF NOT EXISTS prompt_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255),
    content TEXT NOT NULL,              -- The full prompt YAML/text
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Simulation runs (batches of conversations)
CREATE TABLE IF NOT EXISTS simulation_runs (
    id SERIAL PRIMARY KEY,
    version VARCHAR(100) NOT NULL,      -- e.g., "v1.0-baseline"
    prompt_version_id INTEGER REFERENCES prompt_versions(id),
    config JSONB NOT NULL DEFAULT '{}', -- EvaluationConfig as JSON
    status VARCHAR(50) DEFAULT 'pending', -- pending, running, completed, failed
    summary JSONB,                      -- Aggregated results
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Individual conversations
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    simulation_run_id INTEGER REFERENCES simulation_runs(id),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    persona JSONB NOT NULL,             -- Full persona object
    transcript JSONB NOT NULL,          -- Array of turns
    completion_reason VARCHAR(100),
    turn_count INTEGER,
    duration_seconds FLOAT,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Judge configurations (for updateable eval criteria)
CREATE TABLE IF NOT EXISTS judge_configs (
    id SERIAL PRIMARY KEY,
    judge_type VARCHAR(100) NOT NULL,   -- safety, quality, completeness, behavioral
    judge_id VARCHAR(100) NOT NULL,     -- e.g., "safety_crisis_response"
    version VARCHAR(100) NOT NULL,
    config JSONB NOT NULL DEFAULT '{}', -- Judge-specific configuration
    prompt_template TEXT,               -- LLM prompt template if applicable
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(judge_id, version)
);

-- Judgment runs (a set of judgments for a simulation)
CREATE TABLE IF NOT EXISTS judgment_runs (
    id SERIAL PRIMARY KEY,
    simulation_run_id INTEGER REFERENCES simulation_runs(id),
    judge_config_snapshot JSONB,        -- Snapshot of judge configs used
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Individual judgments (evaluation results)
CREATE TABLE IF NOT EXISTS judgments (
    id SERIAL PRIMARY KEY,
    judgment_run_id INTEGER REFERENCES judgment_runs(id),
    conversation_id INTEGER REFERENCES conversations(id),
    judge_type VARCHAR(100) NOT NULL,
    judge_id VARCHAR(100) NOT NULL,
    verdict VARCHAR(50),                -- pass, fail, partial, not_applicable, error
    score FLOAT,
    reasoning TEXT,
    evidence JSONB,                     -- Array of quoted examples
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_simulation_run ON conversations(simulation_run_id);
CREATE INDEX IF NOT EXISTS idx_judgments_conversation ON judgments(conversation_id);
CREATE INDEX IF NOT EXISTS idx_judgments_run ON judgments(judgment_run_id);
CREATE INDEX IF NOT EXISTS idx_simulation_runs_version ON simulation_runs(version);
CREATE INDEX IF NOT EXISTS idx_prompt_versions_version ON prompt_versions(version);
CREATE INDEX IF NOT EXISTS idx_judge_configs_judge_id ON judge_configs(judge_id);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for prompt_versions
DROP TRIGGER IF EXISTS update_prompt_versions_updated_at ON prompt_versions;
CREATE TRIGGER update_prompt_versions_updated_at
    BEFORE UPDATE ON prompt_versions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
