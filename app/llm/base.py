"""
Author: Rongxin rongxin@u.nus.edu
Date: 2025-10-16 16:15:35
LastEditors: Rongxin rongxin@u.nus.edu
LastEditTime: 2025-10-16 16:15:41
FilePath: /agentic-proof-reader/app/llm/base.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Optional

from ..config import settings


class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_content: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout_seconds: Optional[int] = None,
    ) -> str:
        raise NotImplementedError


async def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


def get_provider() -> LLMProvider:
    provider = settings.llm_provider.lower()
    if provider == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider()
    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if provider == "gemini":
        from .gemini_provider import GeminiProvider

        return GeminiProvider()
    if provider == "ollama":
        from .ollama_provider import OllamaProvider

        return OllamaProvider()
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
