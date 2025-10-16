# Agentic Proof Reader (Dev)

Python-based CLI + web GUI using FastAPI, async agents, and pluggable LLMs (OpenAI, Anthropic, Gemini, Ollama). Supports PDF, LaTeX, and Markdown parsing.

## Setup

1. Python 3.10+
2. Create venv and install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Configure env:

```bash
cp .env.example .env
# Fill keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, or set OLLAMA_HOST
# Set LLM_PROVIDER=openai|anthropic|gemini|ollama
```

## Run Web UI

```bash
python -m app.cli serve
# open http://127.0.0.1:8000
```

Upload a `.tex`, `.pdf`, or `.md` file. The server parses content, runs six agents concurrently with a 30s timeout, streams progress over WebSocket, and returns structured results:

```
Original: ...
Problem: ...
Suggestion: ...
Revised version: ... (differences highlighted)
```

## CLI Analyze

```bash
python -m app.cli analyze path/to/file.tex
```

## Notes
- Minimal CDN Vue + Tailwind UI is included under `app/static/index.html`.
- shadcn-vue components can be added later; current UI mimics styles via Tailwind.
- Praison-style async orchestration realized via asyncio tasks and provider abstractions.
