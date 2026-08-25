"""Deep Research - honest scope.

Two modes, chosen automatically based on what's configured:

  - "grounded" mode: if TAVILY_API_KEY is set, each subtopic gets a real
    web search (Tavily's free-tier API) before the LLM writes about it,
    so the answer is actually grounded in current web content.
  - "reasoning" mode: no search key configured - falls back to the LLM's
    own knowledge only, same as the existing ResearchAgent. Still useful,
    but NOT live web research - the response is labeled accordingly so
    callers/UI never claim more than what actually happened.

This mirrors how the big chat products' "Deep Research" works structurally
(break into subtopics, gather evidence per subtopic, synthesize a report)
but is honest about needing a real search API for the "gathers evidence"
part to mean anything beyond the model's training data.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

import requests

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def _extract_json_list(text: str) -> List[str]:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\[.*\])\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
        if bracket_match:
            text = bracket_match.group(0)
    return json.loads(text)


def _plan_subtopics(llm, topic: str, n: int = 4) -> List[str]:
    response = llm.chat(
        [{"role": "user", "content": f"Research topic: {topic}"}],
        system_prompt=(
            f"Break this research topic into {n} focused subtopics/questions "
            'to investigate. Return ONLY a JSON array of strings, e.g. '
            '["subtopic 1", "subtopic 2"]. No other text.'
        ),
        temperature=0.3,
    )
    try:
        subtopics = _extract_json_list(response.text)
        return [s for s in subtopics if isinstance(s, str)][:n]
    except (json.JSONDecodeError, AttributeError):
        return [topic]  # fall back to researching the topic as a single unit


def _web_search(query: str, api_key: str, max_results: int = 4) -> List[Dict[str, str]]:
    resp = requests.post(
        TAVILY_SEARCH_URL,
        json={"api_key": api_key, "query": query, "max_results": max_results},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in data.get("results", [])
    ]


def _research_subtopic_grounded(llm, subtopic: str, api_key: str) -> Dict[str, Any]:
    results = _web_search(subtopic, api_key)
    evidence = "\n\n".join(
        f"[{i+1}] {r['title']} ({r['url']})\n{r['content'][:600]}"
        for i, r in enumerate(results)
    )
    response = llm.chat(
        [{"role": "user", "content": f"Subtopic: {subtopic}\n\nSources:\n{evidence}"}],
        system_prompt=(
            "Summarize what these sources say about the subtopic in 2-4 sentences. "
            "Reference sources by their [n] number where relevant. Stay grounded in "
            "what the sources actually say - don't add outside claims."
        ),
        temperature=0.3,
    )
    return {"subtopic": subtopic, "summary": response.text, "sources": results}


def _research_subtopic_reasoning(llm, subtopic: str) -> Dict[str, Any]:
    response = llm.chat(
        [{"role": "user", "content": f"Subtopic: {subtopic}"}],
        system_prompt=(
            "Give a concise 2-4 sentence answer based on your general knowledge. "
            "This is NOT grounded in live search results - stay appropriately "
            "general and avoid stating specific recent facts/statistics you "
            "can't verify."
        ),
        temperature=0.3,
    )
    return {"subtopic": subtopic, "summary": response.text, "sources": []}


def run_research(llm, topic: str) -> Dict[str, Any]:
    """Returns a report dict. `mode` tells the caller (and should tell the
    UI) whether this was web-grounded or reasoning-only, so nothing gets
    presented as more authoritative than it is."""
    tavily_key = os.environ.get("TAVILY_API_KEY")
    mode = "grounded" if tavily_key else "reasoning"

    subtopics = _plan_subtopics(llm, topic)
    findings = []
    for subtopic in subtopics:
        if mode == "grounded":
            try:
                findings.append(_research_subtopic_grounded(llm, subtopic, tavily_key))
            except requests.RequestException as exc:
                # Search failed for this subtopic (rate limit, network) -
                # degrade to reasoning-only for just this one rather than
                # failing the whole report.
                result = _research_subtopic_reasoning(llm, subtopic)
                result["search_error"] = str(exc)
                findings.append(result)
        else:
            findings.append(_research_subtopic_reasoning(llm, subtopic))

    synthesis_input = "\n\n".join(f"- {f['subtopic']}: {f['summary']}" for f in findings)
    final = llm.chat(
        [{"role": "user", "content": f"Topic: {topic}\n\nFindings:\n{synthesis_input}"}],
        system_prompt="Write a short (3-5 paragraph) synthesized research report from these findings.",
        temperature=0.4,
    )

    return {
        "topic": topic,
        "mode": mode,
        "mode_note": (
            "Grounded in live web search results (Tavily)."
            if mode == "grounded"
            else "Based on the model's general knowledge only - no live web "
                 "search was performed (set TAVILY_API_KEY to enable real "
                 "web-grounded research)."
        ),
        "subtopics": findings,
        "report": final.text,
    }
