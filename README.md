# Daily Read

A 100% local Bible study assistant powered by Ollama and FAISS. Ask questions about the American Standard Version (ASV) Bible and get answers with clickable verse references. Nothing leaves your machine.

## Quick Start

```bash
git clone git@github.com:gramen-isc/asv-rag.git
cd asv-rag
./setup.sh
```

That's it. The setup script installs everything, pulls the AI models, and opens the app in your browser at [localhost:7777](http://localhost:7777).

## Requirements

- **macOS** with [Homebrew](https://brew.sh)
- **Python 3.10+** (`python3 --version` to check)

Everything else is installed automatically by `setup.sh`.

## What `setup.sh` Does

1. Installs [Ollama](https://ollama.com) (local AI runtime) via Homebrew
2. Starts Ollama with a 4-thread CPU limit
3. Downloads two models (~5GB total, one-time):
   - `llama3.2:3b` for answering questions
   - `nomic-embed-text` for searching scripture
4. Creates a Python virtual environment and installs dependencies
5. Launches the web app and opens your browser

## Usage

After the first setup, start the app anytime:

```bash
cd asv-rag
source .venv/bin/activate
python3 web_app.py
```

Or re-run `./setup.sh` which handles everything including starting Ollama.

### Features

- **Chat UI** at localhost:7777 with streaming responses
- **Clickable verse links** that open a Bible page view on the right
- **Hybrid search** combining AI similarity search with topic-aware lookups
- **Conversation memory** for follow-up questions within a session
- **Feedback learning** (thumbs up/down) to improve answers over time
- **Knowledge config** (`knowledge.json`) for teaching the app Bible concepts
- **Terminal panel** showing real-time Ollama activity
- **CLI mode** available via `python3 rag_agent.py`

### Teaching the App

Edit `knowledge.json` to add topics, book groups, or synonyms:

```json
{
  "topics": {
    "your topic": ["Book Chapter", "Book Chapter"]
  },
  "synonyms": {
    "alternate phrase": "existing topic"
  }
}
```

Restart the app to pick up changes. Thumbs-up feedback also teaches the app automatically, saved in `learned_topics.json`.

## Files

| File | Purpose |
|---|---|
| `web_app.py` | Flask web app (main entry point) |
| `rag_agent.py` | CLI chat agent |
| `setup.sh` | One-command installer |
| `asv.txt` | ASV Bible text (~31K verses) |
| `knowledge.json` | Bible topics, book groups, synonyms (editable) |
| `faiss_index/` | Pre-built vector search index |
| `templates/index.html` | Chat UI |
| `requirements.txt` | Python dependencies |
| `learned_topics.json` | Auto-learned topic mappings (from feedback) |
| `qa_cache.json` | Cached answers for instant repeat responses |
