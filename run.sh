#!/usr/bin/env bash
# MemoryAgent — Quick-start server
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

# Check for API key
if [ -z "${QWEN_API_KEY:-}" ]; then
    if [ -f .env ]; then
        set -a; source .env; set +a
    fi
fi

if [ -n "${QWEN_API_KEY:-}" ]; then
    echo "🔑 QwenCloud API key configured — full LLM mode"
else
    echo "⚠️  QWEN_API_KEY not set — running in storage-only mode (no LLM/embeddings)"
    echo "   Copy .env.template to .env and add your key"
fi

echo "🚀 Starting MemoryAgent server on http://localhost:8000"
exec uvicorn memory_agent.main:app --host 0.0.0.0 --port 8000 --reload
