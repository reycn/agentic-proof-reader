from __future__ import annotations

import asyncio
from pathlib import Path

import typer
import uvicorn

from .agents import run_all_agents
from .config import settings
from .parsers.latex_parser import parse_latex
from .parsers.md_parser import parse_markdown
from .parsers.pdf_parser import parse_pdf_bytes
from .server import app

cli = typer.Typer(help="Agentic Proof Reader CLI")


def _parse_path(path: Path) -> str:
    data = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf_bytes(data)
    if suffix == ".tex":
        return parse_latex(data.decode("utf-8", errors="ignore"))
    if suffix in {".md", ".markdown"}:
        return parse_markdown(data.decode("utf-8", errors="ignore"))
    return data.decode("utf-8", errors="ignore")


@cli.command()
def analyze(file: Path) -> None:
    """Analyze a manuscript file and print JSON results."""
    text = _parse_path(file)
    results = asyncio.run(
        run_all_agents(text, timeout_seconds=settings.agent_timeout_seconds)
    )
    import json

    print(json.dumps([r.__dict__ for r in results], ensure_ascii=False, indent=2))


@cli.command()
def serve(
    host: str = typer.Option(settings.host, help="Bind host"),
    port: int = typer.Option(settings.port, help="Bind port"),
) -> None:
    """Run the web server and open the UI."""
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    cli()
