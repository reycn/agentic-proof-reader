"""
Author: Rongxin rongxin@u.nus.edu
Date: 2025-10-16 16:16:59
LastEditors: Rongxin rongxin@u.nus.edu
LastEditTime: 2025-10-16 21:13:47
FilePath: /agentic-proof-reader/app/parsers/pdf_parser.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

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
