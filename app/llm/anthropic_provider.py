from __future__ import annotations

import asyncio
from typing import Optional

import anthropic

from ..config import settings
from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self) -> None:
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

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
            msg = self._client.messages.create(
                model=settings.anthropic_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            # content is a list of TextBlocks
            parts = []
            for block in msg.content:
                if hasattr(block, "text") and block.text:
                    parts.append(block.text)
            return ("\n".join(parts)).strip()

        return await asyncio.wait_for(asyncio.to_thread(_run), timeout=timeout)
