from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, Optional

from .config import settings
from .llm.base import get_provider
from .utils.diff import highlight_differences

try:
    from praisonaiagents import Agent as PraisonAgent
    from praisonaiagents import PraisonAIAgents
    from praisonaiagents import Task as PraisonTask
except Exception:  # noqa: BLE001
    PraisonAgent = None  # type: ignore
    PraisonTask = None  # type: ignore
    PraisonAIAgents = None  # type: ignore


AgentName = Literal[
    "linguistic_polishing",
    "econometric_validation",
    "theorization",
    "precision",
    "logical_reasoning",
    "clarification",
]


@dataclass
class AgentResult:
    name: AgentName
    original: str
    problem: str
    suggestion: str
    revised: str
    highlighted: str


SYSTEM_TEMPLATES: dict[AgentName, str] = {
    "linguistic_polishing": (
        "You are a Linguistic Polishing Agent. Improve clarity, concision, "
        "coherence, and linguistic logic without changing meaning."
    ),
    "econometric_validation": (
        "You are an Econometric Validation Agent. Identify weaknesses in "
        "sampling, model specification, identification, operationalization, "
        "interpretation, presentation, and discussion of results."
    ),
    "theorization": (
        "You are a Theorization Agent. Evaluate paragraph structure, logical "
        "progression, engagement with literature, and integration/paraphrasing "
        "of results."
    ),
    "precision": (
        "You are a Precision Agent. Detect grammatical errors, inconsistent "
        "notations, and contradictory statements."
    ),
    "logical_reasoning": (
        "You are a Logical Reasoning Agent. Identify topic sentences, "
        "sub-arguments, supporting evidence, and assess logical connections; "
        "highlight missing assumptions."
    ),
    "clarification": (
        "You are a Clarification Agent. Flag potential sources of confusion "
        "from concepts, methods, data, or measurements."
    ),
}


def _format_user_prompt(content: str) -> str:
    return (
        "Analyze the following manuscript excerpt and produce a response "
        "exactly in this schema:\n"
        "Original: <paste original>\n"
        "Problem: <identify the problem succinctly>\n"
        "Suggestion: <provide specific actionable suggestion>\n"
        "Revised version: <a fully revised version>; differences highlighted "
        "with <mark> tags\n\n"
        "Text to analyze:\n" + content
    )


ProgressCb = Optional[Callable[[str, str], Awaitable[None]]]


async def run_agent(
    name: AgentName, content: str, *, progress_cb: ProgressCb = None
) -> AgentResult:
    if progress_cb:
        await progress_cb("agent_start", name)
    provider = get_provider()
    system_prompt = SYSTEM_TEMPLATES[name]
    user_prompt = _format_user_prompt(content)
    start = time.perf_counter()
    response = await provider.generate(system_prompt, user_prompt)
    elapsed = time.perf_counter() - start

    original, problem, suggestion, revised = _parse_agent_response(
        response, fallback_original=content[:1000]
    )
    if not original:
        original = content[:1000]
    if not revised:
        revised = original
    highlighted = highlight_differences(original, revised)
    result = AgentResult(
        name=name,
        original=original,
        problem=problem,
        suggestion=suggestion,
        revised=revised,
        highlighted=highlighted,
    )
    if progress_cb:
        await progress_cb("agent_done", f"{name} ({elapsed:.2f}s)")
    return result


async def _run_all_agents_asyncio(
    content: str, *, timeout_seconds: int, progress_cb: ProgressCb
) -> list[AgentResult]:
    names: list[AgentName] = [
        "linguistic_polishing",
        "econometric_validation",
        "theorization",
        "precision",
        "logical_reasoning",
        "clarification",
    ]
    tasks = {
        n: asyncio.create_task(run_agent(n, content, progress_cb=progress_cb))
        for n in names
    }
    done, pending = await asyncio.wait(tasks.values(), timeout=timeout_seconds)

    results: list[AgentResult] = []

    for t in done:
        try:
            results.append(t.result())
        except Exception as exc:  # noqa: BLE001
            name_for_task: AgentName = next(k for k, v in tasks.items() if v is t)
            if progress_cb:
                await progress_cb("agent_error", f"{name_for_task}: {exc}")
            results.append(
                AgentResult(
                    name=name_for_task,
                    original=content[:1000],
                    problem=f"Agent error: {exc}",
                    suggestion="Retry or check provider/API keys.",
                    revised=content[:1000],
                    highlighted=content[:1000],
                )
            )

    for n, t in tasks.items():
        if t in pending:
            t.cancel()
            if progress_cb:
                await progress_cb(
                    "agent_timeout", f"{n}: timeout after {timeout_seconds}s"
                )
            truncated = content[:1000]
            results.append(
                AgentResult(
                    name=n,
                    original=truncated,
                    problem="Agent timed out (30s).",
                    suggestion="Increase timeout or check provider availability.",
                    revised=truncated,
                    highlighted=truncated,
                )
            )

    return results


def _praison_prompt(name: AgentName, content: str) -> str:
    # Use the same strict schema as asyncio path
    instruction = (
        "Analyze the following manuscript excerpt and produce a response exactly in this schema:\n"
        "Original: <paste original>\n"
        "Problem: <identify the problem succinctly>\n"
        "Suggestion: <provide specific actionable suggestion>\n"
        "Revised version: <a fully revised version>; differences highlighted with <mark> tags\n\n"
    )
    return f"{SYSTEM_TEMPLATES[name]}\n\n{instruction}Text to analyze:\n{content}"


async def _run_all_agents_praison(
    content: str, *, timeout_seconds: int, progress_cb: ProgressCb
) -> list[AgentResult]:
    if PraisonAgent is None or PraisonTask is None or PraisonAIAgents is None:
        return await _run_all_agents_asyncio(
            content, timeout_seconds=timeout_seconds, progress_cb=progress_cb
        )

    names: list[AgentName] = [
        "linguistic_polishing",
        "econometric_validation",
        "theorization",
        "precision",
        "logical_reasoning",
        "clarification",
    ]

    results: list[AgentResult] = []

    async def run_single(name: AgentName) -> AgentResult:
        if progress_cb:
            await progress_cb("agent_start", name)
        agent = PraisonAgent(name=name, role="Reviewer", goal="Review text")
        task = PraisonTask(
            description=_praison_prompt(name, content),
            expected_output=(
                "Four labeled fields on separate lines: \n"
                "Original: ...\nProblem: ...\nSuggestion: ...\nRevised version: ..."
            ),
            agent=agent,
        )
        orchestrator = PraisonAIAgents(
            agents=[agent], tasks=[task], process="sequential"
        )
        start = time.perf_counter()

        def _run():
            ret = orchestrator.start()
            # Try multiple places for output
            text = None
            for attr in ("output", "result", "response", "final_output"):
                val = getattr(task, attr, None)
                if val:
                    text = val
                    break
            if text is None:
                for attr in ("results", "output", "result", "responses"):
                    val = getattr(orchestrator, attr, None)
                    if val:
                        text = val
                        break
            if text is None and ret is not None:
                text = ret
            return str(text or "")

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(_run), timeout=timeout_seconds
            )
            elapsed = time.perf_counter() - start
            if progress_cb:
                await progress_cb("agent_done", f"{name} ({elapsed:.2f}s)")
        except asyncio.TimeoutError:
            if progress_cb:
                await progress_cb(
                    "agent_timeout", f"{name}: timeout after {timeout_seconds}s"
                )
            truncated = content[:1000]
            return AgentResult(
                name=name,
                original=truncated,
                problem="Agent timed out (30s).",
                suggestion="Increase timeout or check provider availability.",
                revised=truncated,
                highlighted=truncated,
            )
        except Exception as exc:  # noqa: BLE001
            if progress_cb:
                await progress_cb("agent_error", f"{name}: {exc}")
            truncated = content[:1000]
            return AgentResult(
                name=name,
                original=truncated,
                problem=f"Agent error: {exc}",
                suggestion="Retry or check provider/API keys.",
                revised=truncated,
                highlighted=truncated,
            )

        original = content[:1000]
        original, problem, suggestion, revised = _parse_agent_response(
            response, fallback_original=original
        )
        highlighted = highlight_differences(original, revised)
        return AgentResult(
            name=name,
            original=original,
            problem=problem,
            suggestion=suggestion,
            revised=revised,
            highlighted=highlighted,
        )

    tasks = [run_single(n) for n in names]
    return await asyncio.gather(*tasks)


async def run_all_agents(
    content: str, *, timeout_seconds: int, progress_cb: ProgressCb = None
) -> list[AgentResult]:
    if settings.use_praison:
        return await _run_all_agents_praison(
            content, timeout_seconds=timeout_seconds, progress_cb=progress_cb
        )
    return await _run_all_agents_asyncio(
        content, timeout_seconds=timeout_seconds, progress_cb=progress_cb
    )


def _strip_md(s: str) -> str:
    s = s.strip()
    # remove leading list markers and bold/italics markers
    s = re.sub(r"^\s*[-*]\s*", "", s)
    s = re.sub(r"^\s*(?:\*\*|__)(.*?)(?:\*\*|__)\s*$", r"\1", s)
    return s.strip()


def _parse_agent_response(
    text: str, *, fallback_original: str
) -> tuple[str, str, str, str]:
    # Try JSON first (strip code fences)
    cleaned = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).strip("`"), text)
    candidates = [cleaned, text]
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):

                def g(*keys):
                    for k in keys:
                        for kk in obj.keys():
                            if kk.lower() == k.lower():
                                v = obj[kk]
                                return v if isinstance(v, str) else json.dumps(v)
                    return ""

                original = g("original") or fallback_original
                problem = g("problem")
                suggestion = g("suggestion")
                revised = g("revised version", "revised", "revised_version") or original
                return (
                    original.strip(),
                    problem.strip(),
                    suggestion.strip(),
                    revised.strip(),
                )
        except Exception:
            pass

    # Markdown/label parsing: capture between labels (case-insensitive)
    labels = [
        r"original",
        r"problem",
        r"suggestion",
        r"revised(?:\s+version)?",
    ]
    pattern = re.compile(
        r"(?is)^[ \t]*[-*]?[ \t]*(?:\*\*|__)?(original|problem|suggestion|revised(?:\s+version)?)(?:\*\*|__)?[ \t]*:[ \t]*(.*?)(?=^[ \t]*[-*]?[ \t]*(?:\*\*|__)?(?:"
        + r"|".join(labels)
        + r")(?:\*\*|__)?[ \t]*:|\Z)",
        re.M,
    )
    found = {"original": "", "problem": "", "suggestion": "", "revised version": ""}
    for m in pattern.finditer(text):
        key = m.group(1).lower().replace("  ", " ")
        val = _strip_md(m.group(2))
        if key.startswith("revised"):
            found["revised version"] = val
        else:
            found[key] = val
    original = (found["original"] or fallback_original).strip()
    problem = found["problem"].strip()
    suggestion = found["suggestion"].strip()
    revised = (found["revised version"] or original).strip()
    return original, problem, suggestion, revised
