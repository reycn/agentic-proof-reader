from __future__ import annotations

from difflib import SequenceMatcher


def highlight_differences(
    original: str,
    revised: str,
    *,
    start_tag: str = "<mark>",
    end_tag: str = "</mark>",
) -> str:
    """Return revised string with insertions/replacements highlighted.

    Deletions are ignored in the revised output. Inserted or replaced spans
    are wrapped with start_tag and end_tag.
    """
    matcher = SequenceMatcher(a=original, b=revised)
    out_parts: list[str] = []
    for op, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            out_parts.append(revised[j1:j2])
        elif op in ("insert", "replace"):
            out_parts.append(f"{start_tag}{revised[j1:j2]}{end_tag}")
        elif op == "delete":
            continue
    return "".join(out_parts)
