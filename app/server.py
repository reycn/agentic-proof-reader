from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .agents import run_all_agents
from .config import settings
from .parsers.latex_parser import parse_latex
from .parsers.md_parser import parse_markdown
from .parsers.pdf_parser import parse_pdf_bytes

app = FastAPI(title="Agentic Proof Reader")

# Serve a minimal UI from /static
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root_index() -> HTMLResponse:
    index_file = static_dir / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Agentic Proof Reader</h1><p>UI not found.</p>")
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


def parse_file_bytes(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return parse_pdf_bytes(data)
    if suffix in {".tex"}:
        return parse_latex(data.decode("utf-8", errors="ignore"))
    if suffix in {".md", ".markdown"}:
        return parse_markdown(data.decode("utf-8", errors="ignore"))
    # Fallback: try utf-8
    return data.decode("utf-8", errors="ignore")


# Simple in-memory connections for progress
class ConnectionManager:
    def __init__(self) -> None:
        self.active: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active.discard(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except WebSocketDisconnect:
                self.disconnect(ws)


manager = ConnectionManager()


@app.websocket("/ws/progress")
async def ws_progress(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def report_progress(stage: str, detail: str) -> None:
    await manager.broadcast({"stage": stage, "detail": detail})


@app.get("/api/analyze/self-check")
async def analyze_self_check() -> dict[str, str]:
    # quick check that endpoint responds and orchestration works without file
    await report_progress("self_check", "ok")
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(file: UploadFile) -> JSONResponse:
    async def cb(stage: str, detail: str) -> None:
        await report_progress(stage, detail)

    try:
        await report_progress("upload", f"received {file.filename}")
        data = await file.read()
        await report_progress("parse", "parsing file")
        text = parse_file_bytes(file.filename, data)
        print(text) if text else print("No text found")
        await report_progress("agents", "running six agents")
        results = await run_all_agents(
            text,
            timeout_seconds=settings.agent_timeout_seconds,
            progress_cb=cb,
        )
        await report_progress("done", "completed or timed out")
        return JSONResponse(
            {
                "parsed": text,
                "results": [
                    {
                        "name": r.name,
                        "original": r.original,
                        "problem": r.problem,
                        "suggestion": r.suggestion,
                        "revised": r.revised,
                        "highlighted": r.highlighted,
                    }
                    for r in results
                ],
            }
        )
    except Exception as exc:  # noqa: BLE001
        await report_progress("error", str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
