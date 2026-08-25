"""DATORYX LLM Model - Language model interface and management.

Phase 5: real multi-provider integration. LLMManager routes chat/generate/
stream calls to whichever of Groq, OpenAI, Anthropic, or Gemini is
configured, auto-selecting a provider from the model name (or an explicit
"provider/model" string), retrying transient failures, and falling back to
the next configured provider on a hard failure. If no provider has an API
key configured, it degrades to a local simulated response instead of
crashing - useful for tests, demos, and offline development.
"""
import os
import time
import warnings
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from .providers import (
    PROVIDER_REGISTRY,
    PROVIDER_ENV_KEYS,
    BaseLLMProvider,
    ChatResult,
    LLMProviderError,
)
from .cache import ResponseCache, CacheEntry

MAX_RETRIES_PER_PROVIDER = 2
RETRY_BACKOFF_SECONDS = 1.5


@dataclass
class LLMResponse:
    text: str
    tokens_used: int
    model: str
    latency_ms: float
    finish_reason: str = "stop"
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMManager:
    """Manages LLM interactions across multiple real providers (with offline fallback)."""

    def __init__(self, config: Any = None, auto_register: bool = True, cache: Any = None):
        """
        Args:
            config: optional core.configuration.Configuration instance. When
                given, `llm.default_provider` / `llm.default_model` /
                `llm.temperature` / `llm.max_tokens` are used as defaults.
            auto_register: if True (default), scan the environment for
                GROQ_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY /
                GEMINI_API_KEY and register whichever providers are present.
            cache: optional pre-built ResponseCache to share across manager
                instances. Defaults to a fresh per-manager ResponseCache
                (see models.llm.cache); pass `cache=False` to disable
                caching entirely for this manager.
        """
        self._config = config
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._provider_order: List[str] = []  # registration order = fallback priority
        self._conversation_history: List[Dict] = []
        self._token_usage = {"prompt": 0, "completion": 0}
        self._call_log: List[Dict[str, Any]] = []
        # Exact-match response cache - see models.llm.cache for the full
        # rationale. Sharing one LLMManager instance (the normal case: it's
        # an lru_cache(maxsize=1) singleton behind apps.api.deps) means the
        # cache persists for the life of the process, across every request.
        if cache is False:
            self._cache: Optional[ResponseCache] = ResponseCache(max_entries=0)
        else:
            self._cache = cache if cache is not None else ResponseCache()
        # Optional callback: fn(LLMResponse, meta_dict) -> None, invoked from
        # _finalize() after every completed call (real or simulated). Wired
        # by apps/api/main.py at startup to core.usage.usage_tracker.record,
        # so cost/quota tracking (Phase 16) is purely additive - LLMManager
        # itself has no dependency on core.usage and works unchanged for
        # any non-API caller (CLI, tests) that never sets a hook.
        self._usage_hook: Optional[Any] = None

        self._default_provider = self._cfg("llm.default_provider", "groq")
        self._default_model = self._cfg("llm.default_model", None)
        self._default_temperature = self._cfg("llm.temperature", 0.7)
        self._default_max_tokens = self._cfg("llm.max_tokens", 1000)

        if auto_register:
            self._auto_register_from_env()

    def _cfg(self, key: str, default: Any) -> Any:
        if self._config is not None and hasattr(self._config, "get"):
            return self._config.get(key, default)
        return default

    # ------------------------------------------------------------------
    # Provider registration
    # ------------------------------------------------------------------

    def _auto_register_from_env(self):
        # Cloud providers register first: a deliberately configured API key
        # (GROQ_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY)
        # is an explicit choice and should win the default slot. Local
        # (llama.cpp GGUF) and native (DATORYX's own trained checkpoint) are
        # registered afterwards - they're free, no-network, always-available
        # options, valuable as automatic fallbacks and as an offline/no-key
        # mode, but they must not silently outrank a cloud key the operator
        # actually set. _finalize_default_provider() below is what actually
        # decides the default once every available provider is known.
        for provider_name, env_key in PROVIDER_ENV_KEYS.items():
            api_key = os.environ.get(env_key)
            if api_key:
                try:
                    self.register_provider(provider_name, api_key)
                except Exception as exc:  # pragma: no cover - defensive
                    warnings.warn(f"LLMManager: failed to auto-register '{provider_name}': {exc}")
        self._try_register_local()
        self._try_register_daxro_remote()
        self._try_register_daxro_finetuned()
        self._try_register_native()
        self._finalize_default_provider()

    def _finalize_default_provider(self):
        """Decide which registered provider is actually used when a caller
        doesn't request one explicitly. Priority:

        1. An explicit `llm.default_provider` config value, if that provider
           is actually registered - operator intent always wins.
        2. Any configured cloud provider (a real API key was set) - this is
           a deliberate choice and should not be silently overridden by the
           mere presence of a local/native checkpoint file.
        3. The local llama.cpp GGUF model, if present - still a real,
           capable model, just self-hosted.
        4. DATORYX's own native from-scratch checkpoint - smallest and least
           capable, but always available offline with zero setup.

        This runs once, after every provider has had a chance to register,
        so registration order no longer determines the outcome (previously
        local/native could grab the default slot before cloud providers
        were even registered).
        """
        configured_default = self._cfg("llm.default_provider", None)
        if configured_default and configured_default in self._providers:
            self._default_provider = configured_default
            return

        cloud_names = [p for p in PROVIDER_ENV_KEYS if p in self._providers]
        if cloud_names:
            self._default_provider = (
                self._default_provider if self._default_provider in cloud_names else cloud_names[0]
            )
            return

        if "local" in self._providers:
            self._default_provider = "local"
            return

        if "daxro-remote" in self._providers:
            self._default_provider = "daxro-remote"
            return

        if "daxro" in self._providers:
            self._default_provider = "daxro"
            return

        if "native" in self._providers:
            self._default_provider = "native"
            return

        # Nothing registered at all - leave _default_provider as-is (the
        # "groq" literal default); chat()/generate() already degrade to a
        # clearly-labeled simulated response when self._providers is empty.

    def _try_register_daxro_remote(self):
        """Register the fine-tuned DAXRO model served remotely (e.g. on
        Modal with a GPU - see daxro_modal_deploy/app.py) if
        DAXRO_REMOTE_URL is set. This is the production path for
        DaxroFineTunedProvider's capability on a CPU-only host like Render:
        instead of loading the 3B-parameter model in-process (impossible
        without a local GPU), this just makes an HTTP call to wherever it's
        actually hosted. Ranked above the in-process "daxro" provider and
        the native model, since a working remote endpoint is strictly more
        capable and more likely to actually be reachable on a typical
        deployment than an in-process GPU load.
        """
        from .providers import DaxroRemoteProvider
        url = os.environ.get("DAXRO_REMOTE_URL")
        if not url:
            return  # not configured - fine, other providers still work
        token = os.environ.get("DAXRO_REMOTE_TOKEN")
        try:
            self._providers["daxro-remote"] = DaxroRemoteProvider(url=url, token=token)
            if "daxro-remote" not in self._provider_order:
                insert_at = 1 if "local" in self._provider_order else 0
                self._provider_order.insert(insert_at, "daxro-remote")
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"LLMManager: failed to register daxro-remote: {exc}")

    def _try_register_daxro_finetuned(self):
        """Register the fine-tuned DAXRO model (a real pretrained Qwen2.5-3B
        base + a small LoRA adapter trained on DATORYX-specific data - see
        models/daxro_finetuned/) if an adapter is present. Ranked above the
        tiny from-scratch native model since it's genuinely capable, but
        below cloud providers and the local GGUF model - it requires a GPU
        (see DaxroFineTunedProvider's docstring) and is heavier to load, so
        it's not assumed to be the best default the moment its files exist
        the way a lightweight option would be.
        """
        from .providers import DaxroFineTunedProvider
        adapter_dir = os.path.join(
            os.path.dirname(__file__), "..", "daxro_finetuned", "adapter"
        )
        if not os.path.isfile(os.path.join(adapter_dir, "adapter_model.safetensors")):
            return  # no fine-tuned adapter yet - fine, other providers still work
        try:
            self._providers["daxro"] = DaxroFineTunedProvider(adapter_dir=adapter_dir)
            if "daxro" not in self._provider_order:
                insert_at = 1 if "local" in self._provider_order else 0
                self._provider_order.insert(insert_at, "daxro")
        except ImportError as exc:
            warnings.warn(f"LLMManager: daxro adapter found but transformers/peft/torch isn't installed: {exc}")
        except RuntimeError as exc:
            # Most commonly: no CUDA GPU available. Expected and fine on a
            # CPU-only deployment - just means this provider stays off.
            warnings.warn(f"LLMManager: daxro (fine-tuned) not available: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"LLMManager: failed to load daxro (fine-tuned) model: {exc}")

    def _try_register_native(self):
        """Register DATORYX's own from-scratch trained model (models/native/)
        if a checkpoint exists. Always registered when the checkpoint is
        present, so it's available as a fallback and can be reached
        explicitly (provider="native") - but see _finalize_default_provider()
        for how the *default* provider is actually chosen; a cloud API key
        takes priority over this even though native registers here.
        """
        from .providers import NativeTransformerProvider
        checkpoint_dir = os.path.join(
            os.path.dirname(__file__), "..", "native", "checkpoint"
        )
        if not os.path.isfile(os.path.join(checkpoint_dir, "model.pt")):
            return  # no trained checkpoint yet - fine, other providers still work
        try:
            self._providers["native"] = NativeTransformerProvider(checkpoint_dir=checkpoint_dir)
            if "native" not in self._provider_order:
                insert_at = 1 if "local" in self._provider_order else 0
                self._provider_order.insert(insert_at, "native")
        except ImportError as exc:
            warnings.warn(f"LLMManager: native checkpoint found but torch isn't installed: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"LLMManager: failed to load native model: {exc}")

    def _try_register_local(self):
        from .providers import LocalLlamaProvider
        model_path = os.environ.get("DATORYX_LOCAL_MODEL_PATH")
        if not model_path:
            default_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "local")
            if os.path.isdir(default_dir):
                for fname in os.listdir(default_dir):
                    if fname.endswith(".gguf"):
                        model_path = os.path.join(default_dir, fname)
                        break
        if not model_path or not os.path.isfile(model_path):
            return  # no local model available - fine, cloud providers (if any) still work
        try:
            self._providers["local"] = LocalLlamaProvider(model_path=model_path)
            if "local" not in self._provider_order:
                self._provider_order.insert(0, "local")
        except ImportError as exc:
            warnings.warn(f"LLMManager: local model found but llama-cpp-python isn't installed: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"LLMManager: failed to load local model: {exc}")

    def register_provider(self, name: str, api_key: str,
                           base_url: str = None, models: List[str] = None):
        """Register a real LLM provider by name ('groq', 'openai', 'anthropic', 'gemini')."""
        name = name.lower()
        adapter_cls = PROVIDER_REGISTRY.get(name)
        if adapter_cls is None:
            raise ValueError(
                f"Unknown provider '{name}'. Available: {', '.join(PROVIDER_REGISTRY)}"
            )
        self._providers[name] = adapter_cls(api_key=api_key, base_url=base_url, models=models)
        if name not in self._provider_order:
            self._provider_order.append(name)

    def available_providers(self) -> List[str]:
        return list(self._provider_order)

    def mode(self) -> str:
        """Human-readable summary of what's actually answering chat calls by
        default: 'cloud' (the default provider is Groq/Grok/OpenAI/
        Anthropic/Gemini), 'local' (in-process GGUF via llama.cpp), 'daxro'
        (the fine-tuned Qwen2.5+LoRA model), 'native' (DATORYX's own
        from-scratch trained model), or 'simulated' (nothing registered -
        placeholder text only, not real output).

        Reflects self._default_provider specifically, not merely which
        providers happen to be registered - local/daxro/native are commonly
        registered as fallbacks alongside a cloud provider, and this should
        report what a call with no explicit provider= actually gets, not
        just what's available.
        """
        if not self._providers:
            return "simulated"
        if self._default_provider in PROVIDER_ENV_KEYS:
            return "cloud"
        if self._default_provider in ("local", "daxro-remote", "daxro", "native"):
            return self._default_provider
        return "cloud" if any(p in PROVIDER_ENV_KEYS for p in self._providers) else "simulated"

    def set_usage_hook(self, hook) -> None:
        """Register a callback invoked as hook(LLMResponse, meta_dict) after
        every completed chat()/generate()/stream() call. See core.usage."""
        self._usage_hook = hook

    def is_live(self) -> bool:
        """True if at least one real provider is configured (vs. simulated fallback)."""
        return bool(self._providers)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _resolve_provider_and_model(self, model: Optional[str], provider: Optional[str]):
        """Work out which provider + model to use for this call.

        Accepts either a bare model name ("gpt-4o"), an explicit
        "provider/model" string, or an explicit `provider=` argument.
        Falls back to the configured default provider/model.
        """
        if provider is None and model and "/" in model:
            provider, model = model.split("/", 1)

        if provider is None and model:
            for pname, adapter in self._providers.items():
                if model in adapter.models or model in adapter.known_models:
                    provider = pname
                    break

        if provider is None:
            provider = self._default_provider if self._default_provider in self._providers else (
                self._provider_order[0] if self._provider_order else None
            )

        if model is None:
            model = self._default_model

        return provider, model

    def _fallback_chain(self, preferred: Optional[str]) -> List[str]:
        chain = [preferred] if preferred in self._providers else []
        chain += [p for p in self._provider_order if p not in chain]
        return chain

    # ------------------------------------------------------------------
    # Public API (kept backward compatible with the pre-Phase-5 interface)
    # ------------------------------------------------------------------

    def generate(self, prompt: str, model: str = None,
                 temperature: float = None, max_tokens: int = None,
                 system_prompt: str = None, provider: str = None,
                 response_format: Dict[str, Any] = None) -> LLMResponse:
        """Generate text from a single prompt."""
        return self.chat(
            [{"role": "user", "content": prompt}],
            model=model, temperature=temperature, max_tokens=max_tokens,
            system_prompt=system_prompt, provider=provider, response_format=response_format,
        )

    def chat(self, messages: List[Dict[str, str]], model: str = None,
              temperature: float = None, max_tokens: int = None,
              system_prompt: str = None, provider: str = None,
              response_format: Dict[str, Any] = None, **_ignored) -> LLMResponse:
        """Chat completion with full conversation history sent to the provider."""
        start = time.time()
        temperature = self._default_temperature if temperature is None else temperature
        max_tokens = self._default_max_tokens if max_tokens is None else max_tokens
        self._conversation_history.extend(messages)

        wanted_provider, wanted_model = self._resolve_provider_and_model(model, provider)

        cache_key = None
        if self._cache.is_cacheable(temperature):
            cache_key = self._cache.make_key(messages, wanted_model, wanted_provider,
                                              temperature, max_tokens, system_prompt, response_format)
            cached = self._cache.get(cache_key)
            if cached is not None:
                latency = (time.time() - start) * 1000
                return self._finalize(cached, cached.provider, latency, simulated=False,
                                       raw_model=cached.raw_model, from_cache=True)
        else:
            self._cache.skipped_high_temperature += 1

        if not self._providers:
            result = self._simulate(messages[-1]["content"] if messages else "", system_prompt)
            latency = (time.time() - start) * 1000
            return self._finalize(result, model or "simulated", latency, simulated=True)

        last_error: Optional[Exception] = None
        for pname in self._fallback_chain(wanted_provider):
            adapter = self._providers[pname]
            call_model = wanted_model if pname == wanted_provider else None
            try:
                result = self._call_with_retry(adapter, messages, call_model, temperature,
                                                 max_tokens, system_prompt, response_format)
                latency = (time.time() - start) * 1000
                if cache_key is not None:
                    self._cache.put(cache_key, CacheEntry(
                        text=result.text, prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        finish_reason=result.finish_reason, raw_model=result.raw_model,
                        provider=pname,
                    ))
                return self._finalize(result, pname, latency, simulated=False,
                                       raw_model=result.raw_model)
            except LLMProviderError as exc:
                last_error = exc
                self._call_log.append({"provider": pname, "error": str(exc)})
                continue

        # Every configured provider failed - degrade to simulated rather than crash.
        warnings.warn(f"LLMManager: all providers failed ({last_error}); returning a simulated response")
        result = self._simulate(messages[-1]["content"] if messages else "", system_prompt)
        latency = (time.time() - start) * 1000
        return self._finalize(result, model or "simulated", latency, simulated=True,
                               error=str(last_error) if last_error else None)

    def stream(self, prompt, model: str = None, temperature: float = None,
                max_tokens: int = None, system_prompt: str = None,
                provider: str = None, **_ignored) -> Iterator[str]:
        """Stream tokens as they're generated. Falls back to a single non-streamed
        chunk if no provider is configured or the streaming call fails outright.

        `prompt` accepts either a plain string (wrapped as a single user
        turn, the original behavior) or a full list of {"role", "content"}
        message dicts (multi-turn history) - callers with an ongoing
        conversation (see routers/chat.py) should pass the list so
        streamed replies have the same context as non-streamed chat()
        calls instead of only ever seeing the latest message.
        """
        temperature = self._default_temperature if temperature is None else temperature
        max_tokens = self._default_max_tokens if max_tokens is None else max_tokens
        wanted_provider, wanted_model = self._resolve_provider_and_model(model, provider)
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]

        cache_key = None
        if self._cache.is_cacheable(temperature):
            cache_key = self._cache.make_key(messages, wanted_model, wanted_provider,
                                              temperature, max_tokens, system_prompt)
            cached = self._cache.get(cache_key)
            if cached is not None:
                # Cache hit: nothing to generate, so nothing to wait on. Still
                # yielded word-by-word (rather than one giant chunk) purely so
                # callers that render token-by-token (SSE/WS UI) behave the
                # same as a real stream - it's just instant instead of
                # trickling in at CPU-inference speed.
                for word in cached.text.split():
                    yield word + " "
                return
        else:
            self._cache.skipped_high_temperature += 1

        if not self._providers:
            response = self.chat(messages, model=model, temperature=temperature,
                                  max_tokens=max_tokens, system_prompt=system_prompt)
            for word in response.text.split():
                yield word + " "
            return

        stream_start = time.time()
        for pname in self._fallback_chain(wanted_provider):
            adapter = self._providers[pname]
            call_model = wanted_model if pname == wanted_provider else None
            try:
                yielded_any = False
                streamed_chunks: List[str] = []
                for chunk in adapter.stream(messages, adapter.resolve_model(call_model),
                                             temperature, max_tokens, system_prompt):
                    yielded_any = True
                    streamed_chunks.append(chunk)
                    yield chunk
                if yielded_any:
                    full_text = "".join(streamed_chunks)
                    if cache_key is not None:
                        self._cache.put(cache_key, CacheEntry(
                            text=full_text, prompt_tokens=0, completion_tokens=0,
                            finish_reason="stop", raw_model=adapter.resolve_model(call_model),
                            provider=pname,
                        ))
                    # Provider stream() only yields text chunks, no usage
                    # object - unlike chat(), which goes through _finalize()
                    # and always fires the usage hook. Streamed replies are
                    # the most common billable path (every chat message),
                    # so skipping this here silently gave users free,
                    # unmetered usage on every streamed message. Token
                    # counts aren't available from the stream itself, so
                    # they're estimated the same way the rest of the app
                    # estimates untracked text (~4 chars/token) rather than
                    # left at 0.
                    latency_ms = (time.time() - stream_start) * 1000
                    prompt_chars = sum(len(m.get("content", "")) for m in messages)
                    result = ChatResult(
                        text=full_text,
                        prompt_tokens=max(1, prompt_chars // 4),
                        completion_tokens=max(1, len(full_text) // 4),
                        finish_reason="stop",
                        raw_model=adapter.resolve_model(call_model),
                    )
                    self._finalize(result, pname, latency_ms, simulated=False,
                                    raw_model=result.raw_model)
                    return
            except LLMProviderError:
                continue

        # nothing streamed successfully - fall back to a simulated single chunk
        response = self.chat(messages, model=model, temperature=temperature,
                              max_tokens=max_tokens, system_prompt=system_prompt)
        for word in response.text.split():
            yield word + " "

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call_with_retry(self, adapter: BaseLLMProvider, messages, model, temperature,
                          max_tokens, system_prompt, response_format=None) -> ChatResult:
        attempt = 0
        while True:
            try:
                return adapter.chat(messages, adapter.resolve_model(model), temperature,
                                     max_tokens, system_prompt, response_format=response_format)
            except LLMProviderError as exc:
                attempt += 1
                if not exc.retryable or attempt > MAX_RETRIES_PER_PROVIDER:
                    raise
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    def _finalize(self, result, model_or_provider: str, latency_ms: float, simulated: bool,
                  raw_model: str = None, error: str = None, from_cache: bool = False) -> LLMResponse:
        if simulated:
            text = result
            prompt_tokens = 0
            completion_tokens = len(text.split())
            finish_reason = "stop"
        else:
            text = result.text
            prompt_tokens = result.prompt_tokens
            completion_tokens = result.completion_tokens
            finish_reason = result.finish_reason

        # A cache hit didn't consume any real tokens or compute this call -
        # don't double-count it into usage/billing totals, which already
        # counted these tokens the first time this exact request was made
        # and cached.
        if not from_cache:
            self._token_usage["prompt"] += prompt_tokens
            self._token_usage["completion"] += completion_tokens

        metadata = {
            "simulated": simulated,
            "provider": model_or_provider,
            # Billing/quota tracking (core.usage.tracker.record) reads
            # prompt_tokens/completion_tokens straight off this metadata
            # dict. A cache hit did no real provider call, so it must not
            # be billed or counted against quota - report 0 here even
            # though the *response* below still carries the real token
            # counts for anyone inspecting it out of curiosity.
            "prompt_tokens": 0 if from_cache else prompt_tokens,
            "completion_tokens": 0 if from_cache else completion_tokens,
            "cache_hit": from_cache,
        }
        if raw_model:
            metadata["raw_model"] = raw_model
        if error:
            metadata["fallback_reason"] = error

        response = LLMResponse(
            text=text,
            tokens_used=prompt_tokens + completion_tokens,
            model=raw_model or model_or_provider,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            metadata=metadata,
        )

        if self._usage_hook is not None:
            try:
                self._usage_hook(response, {"simulated": simulated})
            except Exception:  # pragma: no cover - a usage-tracking bug must never break a call
                pass

        return response

    # ------------------------------------------------------------------
    # Offline fallback (no provider configured) - kept intentionally simple.
    # Real generations always go through a registered provider above; this
    # only fires when the manager has zero API keys, so demos/tests/dev
    # environments still work without network access.
    # ------------------------------------------------------------------

    def _simulate(self, prompt: str, system_prompt: str = None) -> str:
        # If the caller asked for JSON-only output (shared.llm_helpers.llm_json
        # always does, via its wrapped system prompt), a prose placeholder
        # would fail json.loads() and get silently swallowed into
        # {"raw_response": ...} - meaning every structured tool (forecasts,
        # extracted entities, etc.) would look "successful" but come back
        # empty with no indication why. Returning valid JSON here instead
        # keeps that failure mode honest and inspectable.
        if system_prompt and "valid json" in system_prompt.lower():
            return json.dumps({
                "simulated": True,
                "note": "No LLM provider configured (GROQ_API_KEY / OPENAI_API_KEY / "
                         "ANTHROPIC_API_KEY / GEMINI_API_KEY) - structured fields are "
                         "intentionally empty rather than fabricated."
            })

        prompt_lower = prompt.lower()
        if "code" in prompt_lower:
            return self._simulate_code(prompt)
        if "explain" in prompt_lower:
            return self._simulate_explanation(prompt)
        if "analyze" in prompt_lower:
            return self._simulate_analysis(prompt)
        return ("[simulated - no LLM provider configured] Based on your request: '"
                + prompt[:50] + "...', here's a placeholder response. Set GROQ_API_KEY, "
                "OPENAI_API_KEY, or ANTHROPIC_API_KEY to get real completions.")

    def _simulate_code(self, prompt: str) -> str:
        return f"""```python
# [simulated - no LLM provider configured]
# Requested: {prompt[:40]}...

def solution():
    result = process_data()
    return result
```"""

    def _simulate_explanation(self, prompt: str) -> str:
        return (f"[simulated - no LLM provider configured]\n## Explanation\n\n"
                f"**Topic:** {prompt[:50]}...\n\nConfigure a real provider "
                f"(GROQ_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY) for an actual answer.")

    def _simulate_analysis(self, prompt: str) -> str:
        return (f"[simulated - no LLM provider configured]\n## Analysis\n\n"
                f"**Subject:** {prompt[:50]}...\n\nConfigure a real provider "
                f"(GROQ_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY) for an actual answer.")

    # ------------------------------------------------------------------

    def get_usage_stats(self) -> Dict[str, Any]:
        return {
            "total_tokens": sum(self._token_usage.values()),
            "prompt_tokens": self._token_usage["prompt"],
            "completion_tokens": self._token_usage["completion"],
            "conversations": len(self._conversation_history) // 2,
            "providers": self.available_providers(),
            "live": self.is_live(),
            "cache": self._cache.stats(),
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Hit/miss counts and current size of the response cache - useful
        on a health/metrics endpoint to see whether caching is actually
        paying off for a given workload."""
        return self._cache.stats()

    def clear_cache(self) -> None:
        self._cache.clear()
