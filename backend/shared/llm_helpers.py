"""DATORYX Shared LLM Helpers.

Small, dependency-free helpers so any module (not just agents.base.BaseAgent
subclasses) can make LLM calls and get back parsed JSON safely. Used by the
brain/ cognitive layer and memory/ subsystems.
"""
from typing import Any, Optional
import json
import re


def safe_float(value: Any, default: float = 0.0,
                minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    """Coerce an untrusted (often LLM-produced) value to a float, falling back
    to `default` instead of raising on garbage input (strings, None, NaN-ish
    text, etc.), then clamp to [minimum, maximum] if given. Used anywhere a
    model's JSON output feeds a numeric field (confidence, duration,
    probability) that downstream code compares or does arithmetic on -
    an out-of-range or non-numeric value should degrade gracefully, not
    crash the caller or silently skew comparisons (e.g. a hallucinated
    confidence of 50 always "winning" against real 0-1 scores)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    if minimum is not None:
        f = max(minimum, f)
    if maximum is not None:
        f = min(maximum, f)
    return f


def safe_dict(value: Any) -> dict:
    """Coerce an untrusted value to a dict, defaulting to {} rather than
    letting a malformed LLM field (e.g. a string or list where an object
    was expected) propagate into code that assumes dict semantics."""
    return value if isinstance(value, dict) else {}


def safe_str_list(value: Any) -> list:
    """Coerce an untrusted value to a list of strings, defaulting to []."""
    if not isinstance(value, list):
        return []
    return [str(v) for v in value]


def truncate_for_prompt(obj: Any, max_chars: int = 4000) -> str:
    """Stringify a value for prompt embedding, capped at max_chars. Guards
    against a caller passing an unexpectedly large context/history blob
    straight into a prompt, which would otherwise inflate token cost and
    can crowd out the instructions themselves."""
    text = str(obj)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [truncated, {len(text) - max_chars} more chars]"


def get_default_llm():
    """Create a default LLMManager. Isolated behind a function to avoid a
    hard import-time dependency for modules that always receive an injected
    llm instance."""
    from models.llm.manager import LLMManager
    return LLMManager()


def llm_text(llm: Any, system_prompt: str, user_prompt: str,
             temperature: float = 0.4, max_tokens: int = 1200) -> str:
    """Run a single-turn completion and return the raw text."""
    response = llm.generate(
        prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.text.strip()


def llm_json(llm: Any, system_prompt: str, user_prompt: str,
             temperature: float = 0.3, max_tokens: int = 1500) -> Any:
    """Run a completion whose system prompt instructs the model to return
    JSON only, then parse it. Tolerates markdown fences and stray prose.

    Two layers of defense against invalid JSON, which matters a lot more on
    a small local model than on GPT-4/Groq-class cloud models:
    1. response_format={"type": "json_object"} is passed to the provider -
       providers that support grammar-constrained generation (llama.cpp
       locally, OpenAI/Groq, Gemini) make invalid JSON syntactically
       impossible rather than just "discouraged by the prompt". Providers
       without an equivalent (Anthropic) just ignore it.
    2. If parsing still fails (e.g. no JSON-mode support, or an
       Anthropic-style provider that only had the prompt instruction to go
       on), one corrective retry is made with the model's own bad output
       shown back to it and asked to fix it - before giving up and falling
       back to a dict wrapping the raw text.
    """
    json_system_prompt = (
        f"{system_prompt}\n\n"
        "Respond with ONLY a single valid JSON object or array. "
        "No markdown code fences, no commentary, no preamble."
    )

    def _try_parse(text: str):
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned), None
        except (json.JSONDecodeError, ValueError):
            match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1)), None
                except (json.JSONDecodeError, ValueError):
                    pass
            return None, cleaned

    response = llm.generate(
        prompt=user_prompt, system_prompt=json_system_prompt,
        temperature=temperature, max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    text = response.text.strip()
    parsed, failed_text = _try_parse(text)
    if parsed is not None:
        return parsed

    # Corrective retry: show the model its own invalid output and ask for a fix.
    retry_prompt = (
        f"{user_prompt}\n\n"
        f"Your previous response was not valid JSON:\n{failed_text[:800]}\n\n"
        "Return ONLY the corrected, valid JSON - nothing else."
    )
    retry_response = llm.generate(
        prompt=retry_prompt, system_prompt=json_system_prompt,
        temperature=min(temperature, 0.2), max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    parsed, _ = _try_parse(retry_response.text.strip())
    if parsed is not None:
        return parsed

    return {"raw_response": text}
