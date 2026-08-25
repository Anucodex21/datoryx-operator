"""DATORYX LLM response cache - Phase 21 (performance).

Exact-match, in-process cache for chat()/stream() results. Local CPU
inference (LocalLlamaProvider, see providers.py) is by far the slowest
link in DATORYX's chat path - a few tokens/second on a laptop with no
GPU - so skipping regeneration entirely for a request DATORYX has already
answered (a repeated demo prompt, an agent re-issuing the same tool-call
system prompt, a user double-clicking "send", a dashboard health check
that asks the same question every N seconds) is often the single biggest
win available without touching hardware or the model file.

Design choices, and why:

- Exact-match only, never semantic/fuzzy. A cache that decides two
  *different* prompts are "close enough" can silently return a wrong or
  stale answer with no visible failure mode - worse than being slow. This
  cache only ever returns a result for the identical (messages, system
  prompt, model/provider, temperature, max_tokens) tuple.
- Off by default for anything but low-temperature calls. High-temperature
  requests are asked *because* the caller wants varied output each time
  (creative writing, brainstorming); serving a cached answer would quietly
  defeat that. Structured/deterministic calls (JSON extraction, routing
  decisions, low-temperature tool prompts - exactly the calls agents make
  over and over) are what this cache is for, and they typically already
  run at temperature 0-0.3.
- Process-local, in-memory, LRU + TTL. No new infra dependency (no Redis
  required); good enough for a single-process API server or a dev/demo
  deployment. If DATORYX is later run with multiple worker processes,
  each gets its own cache - still a net win, no correctness issue.
"""
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CacheEntry:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    raw_model: Optional[str]
    provider: str
    created_at: float = field(default_factory=time.time)


class ResponseCache:
    """Thread-safe LRU + TTL cache of LLM responses, keyed by exact request.

    One instance is shared by an LLMManager across chat() and stream(), so
    a stream() call can be served instantly from a cache entry written by
    an earlier chat() call (or another stream() call) for the same
    request, and vice versa.
    """

    def __init__(self, max_entries: Optional[int] = None, ttl_seconds: Optional[float] = None,
                 max_cacheable_temperature: Optional[float] = None):
        self.max_entries = (
            max_entries if max_entries is not None
            else int(os.environ.get("DATORYX_LLM_CACHE_SIZE", 256))
        )
        self.ttl_seconds = (
            ttl_seconds if ttl_seconds is not None
            else float(os.environ.get("DATORYX_LLM_CACHE_TTL_SECONDS", 3600))
        )
        # Requests hotter than this are never read from or written to the
        # cache - see module docstring. 0.35 comfortably covers "give me a
        # deterministic/structured answer" calls while leaving default
        # (0.7) conversational chat uncached, so a person doesn't get the
        # same phrasing verbatim every time they ask something casually.
        self.max_cacheable_temperature = (
            max_cacheable_temperature if max_cacheable_temperature is not None
            else float(os.environ.get("DATORYX_LLM_CACHE_MAX_TEMPERATURE", 0.35))
        )
        self._store: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.skipped_high_temperature = 0

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0

    def is_cacheable(self, temperature: Optional[float]) -> bool:
        if not self.enabled:
            return False
        if temperature is None:
            return True
        return temperature <= self.max_cacheable_temperature

    @staticmethod
    def make_key(messages: List[Dict[str, str]], model: Optional[str], provider: Optional[str],
                 temperature: Optional[float], max_tokens: Optional[int],
                 system_prompt: Optional[str] = None,
                 response_format: Optional[Dict[str, Any]] = None) -> str:
        payload = {
            "messages": messages,
            "model": model,
            "provider": provider,
            "temperature": round(temperature, 3) if temperature is not None else None,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            "response_format": response_format,
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[CacheEntry]:
        if not self.enabled:
            return None
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            if self.ttl_seconds > 0 and (time.time() - entry.created_at) > self.ttl_seconds:
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)  # LRU: mark as recently used
            self.hits += 1
            return entry

    def put(self, key: str, entry: CacheEntry) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._store[key] = entry
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)  # evict least-recently-used

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0
            self.skipped_high_temperature = 0

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        with self._lock:
            entries = len(self._store)
        return {
            "enabled": self.enabled,
            "entries": entries,
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
            "max_cacheable_temperature": self.max_cacheable_temperature,
            "hits": self.hits,
            "misses": self.misses,
            "skipped_high_temperature": self.skipped_high_temperature,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }
