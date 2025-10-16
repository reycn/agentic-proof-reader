from __future__ import annotations

from io import BytesIO

from markitdown import MarkItDown


def parse_pdf_bytes(data: bytes) -> str:
    md = MarkItDown(enable_plugins=False)
    # convert_stream requires a binary stream; use BytesIO
    result = md.convert_stream(BytesIO(data), file_extension=".pdf")
    # Prefer Markdown text content
    text = getattr(result, "text_content", None) or ""
    return text.strip()
