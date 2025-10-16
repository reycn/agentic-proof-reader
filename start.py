#!/usr/bin/env python3
"""
Simple startup script for the Agentic Proof Reader.
This provides a convenient way to start the server without remembering the module syntax.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import typer

    from app.cli import serve

    # Run the serve command with default options
    typer.run(serve)
    typer.run(serve)
