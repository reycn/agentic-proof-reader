<!--
 * @Author: Rongxin rongxin@u.nus.edu
 * @Date: 2025-10-16 15:59:31
 * @LastEditors: Rongxin rongxin@u.nus.edu
 * @LastEditTime: 2025-10-16 21:59:04
 * @FilePath: /agentic-proof-reader/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# Agentic Proof Reader

Proofreading your manuscript by six agents automatically.

Alpha dev version for personal use & mostly vibe-coded, without commitment for maintenance.

![](./static/screen.png)

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

## Quick Start

### Option 1: Simple startup script (Recommended)
```bash
python start.py
# open http://127.0.0.1:8000
```

### Option 2: Using Make
```bash
make serve
# open http://127.0.0.1:8000
```

### Option 3: Direct CLI
```bash
python -m app.cli serve
# open http://127.0.0.1:8000
```

### Option 4: After installing as package
```bash
pip install -e .
proof-reader-serve
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
# Using the simple script
python start.py --help

# Using Make
make serve-cli

# Direct CLI
python -m app.cli analyze path/to/file.tex

# After installing as package
proof-reader analyze path/to/file.tex
```

## Notes
- Minimal CDN Vue + Tailwind UI is included under `app/static/index.html`.
- shadcn-vue components can be added later; current UI mimics styles via Tailwind.
- Praison-style async orchestration realized via asyncio tasks and provider abstractions.
