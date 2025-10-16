from __future__ import annotations

import re

# rudimentary markdown to text
HEADER = re.compile(r"^\s{0,3}#{1,6}\s+")
CODE_FENCE = re.compile(r"^\s*```")
LINK = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
IMAGE = re.compile(r"!\[[^\]]*\]\([^\)]+\)")
INLINE_CODE = re.compile(r"`([^`]+)`")
HTML_TAG = re.compile(r"<[^>]+>")


def parse_markdown(content: str) -> str:
    out: list[str] = []
    in_code = False
    for line in content.splitlines():
        if CODE_FENCE.match(line):
            in_code = not in_code
            continue
        if in_code:
            continue
        line = IMAGE.sub("", line)
        line = LINK.sub(r"\1", line)
        line = INLINE_CODE.sub(r"\1", line)
        line = HTML_TAG.sub("", line)
        line = HEADER.sub("", line)
        line = re.sub(r"[*_>#-]", " ", line)
        line = re.sub(r"\s+", " ", line)
        if line.strip():
            out.append(line.strip())
    return "\n".join(out)
