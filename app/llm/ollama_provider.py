"""
Author: Rongxin rongxin@u.nus.edu
Date: 2025-10-16 16:16:34
LastEditors: Rongxin rongxin@u.nus.edu
LastEditTime: 2025-10-16 16:16:38
FilePath: /agentic-proof-reader/app/llm/ollama_provider.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

from __future__ import annotations

import asyncio
from typing import Optional

import ollama

from ..config import settings
from .base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self) -> None:
        self._client = ollama.Client(host=settings.ollama_host)

    async def generate(
        self,
        system_prompt: str,
        user_content: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout_seconds: Optional[int] = None,
    ) -> str:
        timeout = timeout_seconds or settings.agent_timeout_seconds

        def _run() -> str:
            resp = self._client.generate(
                model=settings.ollama_model,
                prompt=f"System: {system_prompt}\n\nUser: {user_content}",
                options={"temperature": temperature, "num_predict": max_tokens},
            )
            return (resp.get("response") or "").strip()

        return await asyncio.wait_for(asyncio.to_thread(_run), timeout=timeout)
