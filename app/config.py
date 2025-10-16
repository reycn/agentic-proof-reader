"""
Author: Rongxin rongxin@u.nus.edu
Date: 2025-10-16 16:15:27
LastEditors: Rongxin rongxin@u.nus.edu
LastEditTime: 2025-10-16 18:54:23
FilePath: /agentic-proof-reader/app/config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    host: str = "127.0.0.1"
    port: int = 8001

    # Orchestration
    agent_timeout_seconds: int = 120
    use_praison: bool = False

    # LLM providers
    llm_provider: str = "openai"  # openai | anthropic | gemini | ollama

    openai_api_key: str | None = None
    openai_model: str = "gpt-5-mini"

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-sonnet-20240620"

    google_api_key: str | None = None
    gemini_model: str = "gemini-1.5-pro"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"


settings = Settings()
