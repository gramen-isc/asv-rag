#!/bin/bash
# Setup script for the ASV Bible RAG Agent

set -e

OLLAMA_NUM_THREADS=4
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Daily Read - Setup ==="
echo

# Step 1: Install Ollama
if ! command -v ollama &> /dev/null; then
    echo "[1/5] Installing Ollama..."
    brew install ollama
else
    echo "[1/5] Ollama already installed."
fi

# Step 2: Start Ollama with CPU thread limit
echo "[2/5] Starting Ollama (max ${OLLAMA_NUM_THREADS} CPU threads)..."

# Stop any running Ollama instances
pkill -f "ollama serve" 2>/dev/null || true
sleep 1

if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    export OLLAMA_NUM_THREADS=$OLLAMA_NUM_THREADS
    ollama serve > /tmp/ollama.log 2>&1 &
    echo "  Waiting for Ollama to start..."
    for i in $(seq 1 15); do
        if curl -s http://localhost:11434/api/tags &> /dev/null; then
            break
        fi
        sleep 1
    done
fi

if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "  ERROR: Ollama failed to start. Check /tmp/ollama.log"
    exit 1
fi
echo "  Ollama running (threads=${OLLAMA_NUM_THREADS}, pid=$(pgrep -f 'ollama serve'))."

# Step 3: Pull models
echo "[3/5] Pulling models (this may take a while on first run)..."
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# Step 4: Create venv and install Python dependencies
VENV_DIR="$SCRIPT_DIR/.venv"
echo "[4/5] Setting up Python environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/requirements.txt"

# Step 5: Build FAISS index if it doesn't exist
if [ ! -d "$SCRIPT_DIR/faiss_index" ]; then
    echo "[5/5] Building search index (one-time, may take several minutes)..."
    python3 "$SCRIPT_DIR/rag_agent.py" --build-index 2>&1 || echo "  Index will be built on first run."
else
    echo "[5/5] Search index already exists."
fi

echo
echo "=== Setup complete! ==="
echo

if [ "${1}" = "--no-launch" ]; then
    echo "Start the app:"
    echo "  source .venv/bin/activate"
    echo "  python3 web_app.py"
    exit 0
fi

echo "Launching Daily Read..."
python3 "$SCRIPT_DIR/web_app.py"
