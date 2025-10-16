"""
Author: Rongxin rongxin@u.nus.edu
Date: 2025-10-16 16:16:28
LastEditors: Rongxin rongxin@u.nus.edu
LastEditTime: 2025-10-16 16:16:31
FilePath: /agentic-proof-reader/app/llm/gemini_provider.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

from __future__ import annotations

import asyncio
from typing import Optional

import google.generativeai as genai

from ..config import settings
from .base import LLMProvider


class GeminiProvider(LLMProvider):
    def __init__(self) -> None:
        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        genai.configure(api_key=settings.google_api_key)
        self._model = genai.GenerativeModel(settings.gemini_model)

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
            prompt = f"System: {system_prompt}\n\nUser: {user_content}"
            resp = self._model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            return (resp.text or "").strip()

        return await asyncio.wait_for(asyncio.to_thread(_run), timeout=timeout)
