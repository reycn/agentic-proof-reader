"""
Author: Rongxin rongxin@u.nus.edu
Date: 2025-10-16 16:17:33
LastEditors: Rongxin rongxin@u.nus.edu
LastEditTime: 2025-10-16 20:58:37
FilePath: /agentic-proof-reader/app/agents.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

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
    problem: str
    importance: int  # 0=low, 1=medium, 2=high
    location: str
    suggestion_brief: str
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
        "Problem: <identify the problem succinctly>\n"
        "Importance: <1|2|3|4|5|6|7|8|9|10 where 1=lowest, 10=highest>\n"
        "Location of problem: <quote up to 2 sentences, max 500 chars>\n"
        "Suggestion (brief): <one or two sentences>\n"
        "Revised: <suggested improved sentence(s), truncate to 500 chars>\n\n"
        "Text to analyze:\n" + content
    )


ProgressCb = Optional[Callable[[str, str], Awaitable[None]]]
ResultCb = Optional[Callable[["AgentResult"], Awaitable[None]]]


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

    (
        problem,
        importance,
        location,
        suggestion_brief,
        revised,
    ) = _parse_agent_response(response)

    highlighted = highlight_differences(location or content[:200], revised or location)
    result = AgentResult(
        name=name,
        problem=problem,
        importance=importance,
        location=location,
        suggestion_brief=suggestion_brief,
        revised=revised,
        highlighted=highlighted,
    )
    if progress_cb:
        await progress_cb("agent_done", f"{name} ({elapsed:.2f}s)")
    return result


async def _run_all_agents_asyncio(
    content: str,
    *,
    timeout_seconds: int,
    progress_cb: ProgressCb,
    result_cb: ResultCb,
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
            r = t.result()
            results.append(r)
            if result_cb:
                await result_cb(r)
        except Exception as exc:  # noqa: BLE001
            name_for_task: AgentName = next(k for k, v in tasks.items() if v is t)
            if progress_cb:
                await progress_cb("agent_error", f"{name_for_task}: {exc}")
            r = AgentResult(
                name=name_for_task,
                problem=f"Agent error: {exc}",
                importance=5,
                location="",
                suggestion_brief="Retry or check provider/API keys.",
                revised="",
                highlighted="",
            )
            results.append(r)
            if result_cb:
                await result_cb(r)

    for n, t in tasks.items():
        if t in pending:
            t.cancel()
            if progress_cb:
                await progress_cb(
                    "agent_timeout", f"{n}: timeout after {timeout_seconds}s"
                )
            r = AgentResult(
                name=n,
                problem="Agent timed out.",
                importance=5,
                location="",
                suggestion_brief="Increase timeout or check provider availability.",
                revised="",
                highlighted="",
            )
            results.append(r)
            if result_cb:
                await result_cb(r)

    return results


def _praison_prompt(name: AgentName, content: str) -> str:
    instruction = (
        "Analyze the following manuscript excerpt and produce a response exactly in this schema:\n"
        "Problem: <identify the problem succinctly>\n"
        "Importance: <1|2|3|4|5|6|7|8|9|10 where 1=lowest, 10=highest>\n"
        "Location of problem: <quote up to 2 sentences, max 500 chars>\n"
        "Suggestion (brief): <one or two sentences>\n"
        "Revised: <suggested improved sentence(s), truncate to 500 chars>\n\n"
    )
    return f"{SYSTEM_TEMPLATES[name]}\n\n{instruction}Text to analyze:\n{content}"


async def _run_all_agents_praison(
    content: str,
    *,
    timeout_seconds: int,
    progress_cb: ProgressCb,
    result_cb: ResultCb,
) -> list[AgentResult]:
    if PraisonAgent is None or PraisonTask is None or PraisonAIAgents is None:
        return await _run_all_agents_asyncio(
            content,
            timeout_seconds=timeout_seconds,
            progress_cb=progress_cb,
            result_cb=result_cb,
        )

    names: list[AgentName] = [
        "linguistic_polishing",
        "econometric_validation",
        "theorization",
        "precision",
        "logical_reasoning",
        "clarification",
    ]

    async def run_single(name: AgentName) -> AgentResult:
        if progress_cb:
            await progress_cb("agent_start", name)
        agent = PraisonAgent(name=name, role="Reviewer", goal="Review text")
        task = PraisonTask(
            description=_praison_prompt(name, content),
            expected_output=(
                "Five labeled fields on separate lines:\n"
                "Problem: ...\n"
                "Importance: 1|2|3|4|5|6|7|8|9|10\n"
                "Location of problem: ...\n"
                "Suggestion (brief): ...\n"
                "Revised: ..."
            ),
            agent=agent,
        )
        orchestrator = PraisonAIAgents(
            agents=[agent], tasks=[task], process="sequential"
        )
        start = time.perf_counter()

        def _run():
            ret = orchestrator.start()
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
            r = AgentResult(
                name=name,
                problem="Agent timed out.",
                importance=5,
                location="",
                suggestion_brief="Increase timeout or check provider availability.",
                revised="",
                highlighted="",
            )
            if result_cb:
                await result_cb(r)
            return r
        except Exception as exc:  # noqa: BLE001
            if progress_cb:
                await progress_cb("agent_error", f"{name}: {exc}")
            r = AgentResult(
                name=name,
                problem=f"Agent error: {exc}",
                importance=5,
                location="",
                suggestion_brief="Retry or check provider/API keys.",
                revised="",
                highlighted="",
            )
            if result_cb:
                await result_cb(r)
            return r

        problem, importance, location, suggestion_brief, revised = (
            _parse_agent_response(response)
        )
        highlighted = highlight_differences(
            location or content[:200], revised or location
        )
        r = AgentResult(
            name=name,
            problem=problem,
            importance=importance,
            location=location,
            suggestion_brief=suggestion_brief,
            revised=revised,
            highlighted=highlighted,
        )
        if result_cb:
            await result_cb(r)
        return r

    tasks = [run_single(n) for n in names]
    return await asyncio.gather(*tasks)


async def run_all_agents(
    content: str,
    *,
    timeout_seconds: int,
    progress_cb: ProgressCb = None,
    result_cb: ResultCb = None,
) -> list[AgentResult]:
    if settings.use_praison:
        return await _run_all_agents_praison(
            content,
            timeout_seconds=timeout_seconds,
            progress_cb=progress_cb,
            result_cb=result_cb,
        )
    return await _run_all_agents_asyncio(
        content,
        timeout_seconds=timeout_seconds,
        progress_cb=progress_cb,
        result_cb=result_cb,
    )


@dataclass
class ChunkTask:
    chunk_id: int
    agent_name: AgentName
    content: str


@dataclass
class ChunkTaskResult:
    chunk_id: int
    agent_name: AgentName
    result: AgentResult


async def run_all_agents_distributed(
    content: str,
    *,
    timeout_seconds: int,
    progress_cb: ProgressCb = None,
    result_cb: ResultCb = None,
) -> list[ChunkTaskResult]:
    """
    Run all 6 agents on each paragraph chunk of the content.
    Creates m chunks × 6 agents = m×6 total tasks.
    """
    # Split content into paragraphs
    chunks = chunk_by_paragraphs(content)

    if progress_cb:
        await progress_cb("chunking", f"Split into {len(chunks)} paragraphs")

    # Create all tasks (m chunks × 6 agents)
    agent_names: list[AgentName] = [
        "linguistic_polishing",
        "econometric_validation",
        "theorization",
        "precision",
        "logical_reasoning",
        "clarification",
    ]

    all_tasks: list[ChunkTask] = []
    for chunk_id, chunk_content in enumerate(chunks):
        for agent_name in agent_names:
            all_tasks.append(
                ChunkTask(
                    chunk_id=chunk_id, agent_name=agent_name, content=chunk_content
                )
            )

    if progress_cb:
        await progress_cb(
            "task_creation",
            f"Created {len(all_tasks)} tasks ({len(chunks)} chunks × {len(agent_names)} agents)",
        )

    # Run all tasks asynchronously
    async def run_chunk_task(task: ChunkTask) -> ChunkTaskResult:
        if progress_cb:
            await progress_cb(
                "task_start", f"Chunk {task.chunk_id + 1} - {task.agent_name}"
            )

        try:
            result = await run_agent(
                task.agent_name,
                task.content,
                progress_cb=None,  # Don't pass progress_cb to individual agents to avoid spam
            )

            chunk_result = ChunkTaskResult(
                chunk_id=task.chunk_id, agent_name=task.agent_name, result=result
            )

            if progress_cb:
                await progress_cb(
                    "task_done", f"Chunk {task.chunk_id + 1} - {task.agent_name}"
                )

            if result_cb:
                await result_cb(chunk_result)

            return chunk_result

        except Exception as exc:
            if progress_cb:
                await progress_cb(
                    "task_error",
                    f"Chunk {task.chunk_id + 1} - {task.agent_name}: {exc}",
                )

            # Create error result
            error_result = AgentResult(
                name=task.agent_name,
                problem=f"Task error: {exc}",
                importance=5,
                location="",
                suggestion_brief="Retry or check provider availability.",
                revised="",
                highlighted="",
            )

            chunk_result = ChunkTaskResult(
                chunk_id=task.chunk_id, agent_name=task.agent_name, result=error_result
            )

            if result_cb:
                await result_cb(chunk_result)

            return chunk_result

    # Execute all tasks with timeout
    try:
        tasks = [run_chunk_task(task) for task in all_tasks]
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=timeout_seconds
        )

        # Filter out exceptions and convert to ChunkTaskResult
        chunk_results: list[ChunkTaskResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if progress_cb:
                    await progress_cb("task_exception", f"Task {i}: {result}")
                # Create error result
                task = all_tasks[i]
                error_result = AgentResult(
                    name=task.agent_name,
                    problem=f"Task exception: {result}",
                    importance=5,
                    location="",
                    suggestion_brief="Check system logs.",
                    revised="",
                    highlighted="",
                )
                chunk_results.append(
                    ChunkTaskResult(
                        chunk_id=task.chunk_id,
                        agent_name=task.agent_name,
                        result=error_result,
                    )
                )
            else:
                chunk_results.append(result)

        if progress_cb:
            await progress_cb("all_done", f"Completed {len(chunk_results)} tasks")

        return chunk_results

    except asyncio.TimeoutError:
        if progress_cb:
            await progress_cb("timeout", f"Overall timeout after {timeout_seconds}s")

        # Return whatever results we have
        return []


def chunk_by_paragraphs(content: str) -> list[str]:
    """
    Split content into real, continuous paragraphs using intelligent detection.

    This function identifies actual paragraphs by looking for:
    1. Double line breaks (traditional paragraph breaks)
    2. Sentence endings followed by line breaks
    3. Content structure patterns (indentation, formatting)
    4. Minimum paragraph length and coherence
    """
    if not content.strip():
        return []

    # Clean up the content first
    content = content.strip()

    # Method 1: Split by explicit paragraph breaks (double newlines)
    explicit_paragraphs = re.split(r"\n\s*\n+", content)
    paragraphs = []

    for para in explicit_paragraphs:
        para = para.strip()
        if _is_valid_paragraph(para):
            paragraphs.append(para)

    # Method 2: If we have few paragraphs, try more sophisticated detection
    if len(paragraphs) <= 1:
        paragraphs = _detect_paragraphs_by_structure(content)

    # Method 3: If still no good paragraphs, use sentence-based chunking
    if len(paragraphs) <= 1:
        paragraphs = _detect_paragraphs_by_sentences(content)

    # Filter and clean up paragraphs
    valid_paragraphs = []
    for para in paragraphs:
        para = _clean_paragraph(para)
        if _is_valid_paragraph(para):
            valid_paragraphs.append(para)

    return valid_paragraphs if valid_paragraphs else [content]


def _is_valid_paragraph(text: str) -> bool:
    """Check if text qualifies as a valid paragraph."""
    if not text or len(text.strip()) < 50:
        return False

    # Must have at least one sentence (contains sentence-ending punctuation)
    if not re.search(r"[.!?]", text):
        return False

    # Should not be just formatting or metadata
    if re.match(r"^[#*\-+\s\d\.]+$", text):
        return False

    # Should contain actual words (not just symbols/numbers)
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    if len(words) < 5:
        return False

    return True


def _clean_paragraph(text: str) -> str:
    """Clean up a paragraph by removing excessive whitespace and formatting."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove excessive punctuation
    text = re.sub(r"[.]{3,}", "...", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def _detect_paragraphs_by_structure(content: str) -> list[str]:
    """Detect paragraphs based on structural patterns."""
    lines = content.split("\n")
    paragraphs = []
    current_para = []

    for line in lines:
        line = line.strip()
        if not line:
            # Empty line - potential paragraph break
            if current_para:
                para_text = " ".join(current_para)
                if _is_valid_paragraph(para_text):
                    paragraphs.append(para_text)
                current_para = []
        else:
            # Check for paragraph indicators
            is_new_paragraph = False

            # Indented lines might start new paragraphs
            if line.startswith("    ") or line.startswith("\t"):
                if current_para:
                    para_text = " ".join(current_para)
                    if _is_valid_paragraph(para_text):
                        paragraphs.append(para_text)
                    current_para = []
                is_new_paragraph = True

            # Lines starting with numbers/bullets might be new paragraphs
            elif re.match(r"^\d+[\.\)]\s", line) or re.match(r"^[-*+]\s", line):
                if current_para:
                    para_text = " ".join(current_para)
                    if _is_valid_paragraph(para_text):
                        paragraphs.append(para_text)
                    current_para = []
                is_new_paragraph = True

            # Very long lines might be separate paragraphs
            elif current_para and len(line) > 200:
                para_text = " ".join(current_para)
                if _is_valid_paragraph(para_text):
                    paragraphs.append(para_text)
                current_para = [line]
                is_new_paragraph = True

            if not is_new_paragraph:
                current_para.append(line)

    # Add the last paragraph
    if current_para:
        para_text = " ".join(current_para)
        if _is_valid_paragraph(para_text):
            paragraphs.append(para_text)

    return paragraphs


def _detect_paragraphs_by_sentences(content: str) -> list[str]:
    """Detect paragraphs based on sentence patterns."""
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", content)

    paragraphs = []
    current_para = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        current_para.append(sentence)

        # Create paragraph when we have enough sentences or reach natural breaks
        if len(current_para) >= 3:  # Minimum 3 sentences per paragraph
            para_text = " ".join(current_para)
            if _is_valid_paragraph(para_text):
                paragraphs.append(para_text)
                current_para = []

    # Add remaining sentences as final paragraph
    if current_para:
        para_text = " ".join(current_para)
        if _is_valid_paragraph(para_text):
            paragraphs.append(para_text)

    return paragraphs


def _strip_md(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\s*[-*]\s*", "", s)
    s = re.sub(r"^\s*(?:\*\*|__)(.*?)(?:\*\*|__)\s*$", r"\1", s)
    return s.strip()


def _parse_agent_response(text: str) -> tuple[str, int, str, str, str]:
    # JSON attempt
    cleaned = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).strip("`"), text)
    for cand in (cleaned, text):
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

                problem = g("problem")
                imp_raw = g("importance") or "5"
                # Accept numeric or textual
                try:
                    importance = int(str(imp_raw).strip())
                except Exception:
                    txt = str(imp_raw).lower().strip()
                    # Map old textual values to new scale
                    importance = {
                        "low": 3,
                        "lowest": 1,
                        "medium": 5,
                        "high": 8,
                        "highest": 10,
                    }.get(txt, 5)
                location = g(
                    "location of problem",
                    "location",
                    "problem location",
                    "problematic sentences",
                    "problematic",
                )
                sugg = g("suggestion (brief)", "suggestion", "suggestion_brief")
                revised = g("revised", "revised sentences")
                importance = max(1, min(10, importance))
                return (
                    problem.strip(),
                    importance,
                    location.strip(),
                    sugg.strip(),
                    revised.strip(),
                )
        except Exception:
            pass

    # Label parsing
    labels = [
        r"problem",
        r"importance",
        r"location(?:\s+of\s+problem)?|problem(?:\s+location)?|problematic(?:\s+sentences?)?",
        r"suggestion(?:\s*\(brief\))?",
        r"revised(?:\s+sentences?)?",
    ]
    pattern = re.compile(
        r"(?is)^[ \t]*[-*]?[ \t]*(?:\*\*|__)?(problem|importance|location(?:\s+of\s+problem)?|problem(?:\s+location)?|problematic(?:\s+sentences?)?|suggestion(?:\s*\(brief\))?|revised(?:\s+sentences?)?)(?:\*\*|__)?[ \t]*:[ \t]*(.*?)(?=^[ \t]*[-*]?[ \t]*(?:\*\*|__)?(?:"
        + r"|".join(labels)
        + r")(?:\*\*|__)?[ \t]*:|\Z)",
        re.M,
    )
    found = {
        "problem": "",
        "importance": "",
        "location of problem": "",
        "suggestion (brief)": "",
        "revised": "",
    }
    for m in pattern.finditer(text):
        key = m.group(1).lower()
        val = _strip_md(m.group(2))
        if (
            key.startswith("location")
            or key.startswith("problem ")
            or key.startswith("problematic")
        ):
            found["location of problem"] = val
        elif key.startswith("suggestion"):
            found["suggestion (brief)"] = val
        elif key.startswith("revised"):
            found["revised"] = val
        else:
            found[key] = val
    problem = found["problem"].strip()
    imp_raw = found["importance"].strip() or "5"
    try:
        importance = int(imp_raw)
    except Exception:
        txt = imp_raw.lower()
        importance = {"low": 3, "lowest": 1, "medium": 5, "high": 8, "highest": 10}.get(
            txt, 5
        )
    importance = max(1, min(10, importance))
    location = found["location of problem"].strip()
    sugg = found["suggestion (brief)"].strip()
    revised = found["revised"].strip()
    return problem, importance, location, sugg, revised
