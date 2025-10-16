from __future__ import annotations

import re
from typing import Iterable

LATEX_COMMAND_PATTERN = re.compile(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?")
ENV_BEGIN = re.compile(r"\\begin\{([^}]*)\}")
ENV_END = re.compile(r"\\end\{([^}]*)\}")


def _strip_comments(line: str) -> str:
    # Remove % comments not escaped
    return re.sub(r"(?<!\\)%.*$", "", line)


def _remove_commands(text: str) -> str:
    # Remove common LaTeX commands and keep arguments' content when possible
    text = re.sub(r"\\cite[t|p]?\{[^}]*\}", "", text)
    text = re.sub(r"\\ref\{[^}]*\}", "", text)
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\\footnote\{([^}]*)\}", r" (footnote: \1) ", text)
    # Remove other commands entirely
    text = LATEX_COMMAND_PATTERN.sub("", text)
    # Remove math environments inline
    text = re.sub(r"\$\$[\s\S]*?\$\$", " ", text)
    text = re.sub(r"\$[^$]*\$", " ", text)
    # Normalize spaces
    return re.sub(r"\s+", " ", text).strip()


def parse_latex(content: str) -> str:
    lines: Iterable[str] = content.splitlines()
    cleaned: list[str] = []
    stack: list[str] = []
    for raw in lines:
        line = _strip_comments(raw)
        m_begin = ENV_BEGIN.search(line)
        m_end = ENV_END.search(line)
        if m_begin:
            stack.append(m_begin.group(1))
        if m_end and stack:
            stack.pop()
        if not line.strip():
            continue
        cleaned.append(_remove_commands(line))
    return "\n".join(s for s in cleaned if s)
