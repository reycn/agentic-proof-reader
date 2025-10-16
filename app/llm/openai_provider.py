from __future__ import annotations

import asyncio
from typing import Optional

from openai import OpenAI

from ..config import settings
from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=settings.openai_api_key)

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
            response = self._client.chat.completions.create(
                model=settings.openai_model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            return (response.choices[0].message.content or "").strip()

        return await asyncio.wait_for(asyncio.to_thread(_run), timeout=timeout)
