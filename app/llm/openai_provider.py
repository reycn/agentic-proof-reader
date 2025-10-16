"""
Author: Rongxin rongxin@u.nus.edu
Date: 2025-10-16 16:16:12
LastEditors: Rongxin rongxin@u.nus.edu
LastEditTime: 2025-10-16 19:28:50
FilePath: /agentic-proof-reader/app/llm/openai_provider.py
Description: Default settings, please set `customMade`, open koroFileHeader
# to view configuration: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

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
        max_tokens: int = 1024,
        timeout_seconds: Optional[int] = None,
    ) -> str:
        timeout = timeout_seconds or settings.agent_timeout_seconds

        def _run() -> str:
            # Determine the correct parameter name based on the model
            model_name = settings.openai_model.lower()

            # Models that use max_completion_tokens instead of max_tokens
            completion_token_models = [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
            ]

            # Check if the model uses max_completion_tokens
            use_completion_tokens = any(
                model_name.startswith(prefix) for prefix in completion_token_models
            )

            # Prepare the parameters
            params = {
                "model": settings.openai_model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            }

            # Use the correct parameter name
            if use_completion_tokens:
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens

            response = self._client.chat.completions.create(**params)
            return (response.choices[0].message.content or "").strip()

        return await asyncio.wait_for(asyncio.to_thread(_run), timeout=timeout)
