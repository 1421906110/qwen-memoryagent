#!/bin/bash
# CogniMem Docker entrypoint
# Starts both engine (8001) and UI (8000)

echo "🚀 Starting CogniMem..."

# Start engine in background
uvicorn cognimem.main:app --host 0.0.0.0 --port 8001 &
ENGINE_PID=$!
echo "🧠 Engine starting on :8001 (PID $ENGINE_PID)"

# Wait for engine
for i in $(seq 1 15); do
    curl -s -o /dev/null http://localhost:8001/ && break
    sleep 1
done

# Start UI in foreground
echo "💬 UI starting on :8000"
exec uvicorn memory_agent.main:app --host 0.0.0.0 --port 8000
