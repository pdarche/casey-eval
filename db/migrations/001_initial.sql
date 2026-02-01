-- Migration 001: Initial schema
-- This migration creates all tables needed for the ODL evaluation system

-- Run the main schema
\i /docker-entrypoint-initdb.d/01-schema.sql

-- Insert migration record
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(100) PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO schema_migrations (version) VALUES ('001_initial')
ON CONFLICT (version) DO NOTHING;
