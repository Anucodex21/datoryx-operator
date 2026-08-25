"""DATORYX LLM Providers - Thin HTTP adapters over real provider APIs.

Each provider implements the same small surface (chat / stream / usage
parsing) so LLMManager can treat Groq, OpenAI, Anthropic, and Gemini
interchangeably. Groq and OpenAI speak the same wire format (OpenAI's
chat-completions schema), so they share `OpenAICompatibleProvider`.
Anthropic and Gemini have their own schemas and get dedicated adapters.
"""
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import requests

DEFAULT_TIMEOUT = 60


def _env_int(name: str, default: int) -> int:
    """Read an int from an env var, treating unset OR empty-string as
    "use the default" instead of crashing.

    A generated .env (see scripts/setup_local.py) can legitimately write
    `SOME_VAR=` for a setting the user hasn't customized - that's still a
    "present but empty" value, not a valid int, and plain `int(os.environ
    .get(name, default))` blows up on it because `.get()` only falls back
    to `default` when the key is fully absent, not when it's set to "".
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


class LLMProviderError(Exception):
    """Raised when a provider call fails (auth, rate limit, network, bad response)."""

    def __init__(self, provider: str, message: str, status_code: Optional[int] = None, retryable: bool = False):
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        super().__init__(f"[{provider}] {message}")


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    raw_model: str


class BaseLLMProvider(ABC):
    """Common interface every provider adapter implements."""

    #: registry name, e.g. "groq"
    name: str = "base"
    #: model used when the caller doesn't specify one
    default_model: str = ""
    #: model name prefixes/exact names this provider recognizes, for auto-routing
    known_models: tuple = ()

    def __init__(self, api_key: str, base_url: Optional[str] = None, models: Optional[List[str]] = None):
        if not api_key:
            raise ValueError(f"{self.name}: an API key is required")
        self.api_key = api_key
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.models = models or list(self.known_models)

    @property
    @abstractmethod
    def default_base_url(self) -> str:
        ...

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], model: str, temperature: float,
              max_tokens: int, system_prompt: Optional[str] = None,
              response_format: Optional[Dict[str, Any]] = None) -> ChatResult:
        """Send a synchronous chat completion request and return the parsed result.

        response_format: optional, e.g. {"type": "json_object"}. Providers that
        support constraining generation to valid JSON (OpenAI/Groq's JSON mode,
        Gemini's responseMimeType, llama.cpp's GBNF JSON grammar) should honor
        it; providers that don't (Anthropic has no equivalent) should just
        ignore it rather than error - the caller still gets a best-effort
        response and validates/parses on its own.
        """

    @abstractmethod
    def stream(self, messages: List[Dict[str, str]], model: str, temperature: float,
                max_tokens: int, system_prompt: Optional[str] = None) -> Iterator[str]:
        """Yield text chunks as they arrive from the provider."""

    def resolve_model(self, model: Optional[str]) -> str:
        return model or self.default_model

    def _headers(self) -> Dict[str, str]:
        raise NotImplementedError

    @staticmethod
    def _raise_for_status(provider: str, response: "requests.Response"):
        if response.ok:
            return
        retryable = response.status_code == 429 or response.status_code >= 500
        try:
            detail = response.json()
            message = detail.get("error", {}).get("message") or detail.get("message") or response.text
        except ValueError:
            message = response.text
        raise LLMProviderError(provider, message[:300], status_code=response.status_code, retryable=retryable)


class OpenAICompatibleProvider(BaseLLMProvider):
    """Shared implementation for any provider that speaks OpenAI's chat-completions API.

    Used directly for OpenAI, and subclassed (with a different base URL and
    default model) for Groq, which is fully OpenAI-wire-compatible.
    """

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_messages(self, messages: List[Dict[str, str]], system_prompt: Optional[str]) -> List[Dict[str, str]]:
        payload = list(messages)
        if system_prompt and not any(m.get("role") == "system" for m in payload):
            payload = [{"role": "system", "content": system_prompt}] + payload
        return payload

    def chat(self, messages, model, temperature, max_tokens, system_prompt=None,
              response_format=None) -> ChatResult:
        model = self.resolve_model(model)
        body = {
            "model": model,
            "messages": self._build_messages(messages, system_prompt),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            body["response_format"] = response_format
        resp = requests.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                              json=body, timeout=DEFAULT_TIMEOUT)
        self._raise_for_status(self.name, resp)
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return ChatResult(
            text=choice["message"]["content"] or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
            raw_model=data.get("model", model),
        )

    def stream(self, messages, model, temperature, max_tokens, system_prompt=None) -> Iterator[str]:
        model = self.resolve_model(model)
        body = {
            "model": model,
            "messages": self._build_messages(messages, system_prompt),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        with requests.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                            json=body, timeout=DEFAULT_TIMEOUT, stream=True) as resp:
            self._raise_for_status(self.name, resp)
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                text = delta.get("content")
                if text:
                    yield text


class GroqProvider(OpenAICompatibleProvider):
    name = "groq"
    default_model = "llama-3.3-70b-versatile"
    known_models = (
        "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
        "mixtral-8x7b-32768", "gemma2-9b-it", "deepseek-r1-distill-llama-70b",
    )

    @property
    def default_base_url(self) -> str:
        return "https://api.groq.com/openai/v1"


class GrokProvider(OpenAICompatibleProvider):
    """xAI's Grok - not to be confused with Groq (above), which is a
    separate company/API that happens to have a near-identical name. Grok
    is fully OpenAI-wire-compatible, same as Groq, so this just points
    OpenAICompatibleProvider at xAI's endpoint."""
    name = "grok"
    default_model = "grok-4"
    known_models = (
        "grok-4", "grok-4-fast", "grok-3", "grok-3-mini",
    )

    @property
    def default_base_url(self) -> str:
        return "https://api.x.ai/v1"


class OpenAIProvider(OpenAICompatibleProvider):
    name = "openai"
    default_model = "gpt-4o-mini"
    known_models = ("gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "o1", "o1-mini", "o3-mini")

    @property
    def default_base_url(self) -> str:
        return "https://api.openai.com/v1"


class AnthropicProvider(BaseLLMProvider):
    name = "anthropic"
    default_model = "claude-sonnet-4-6"
    known_models = (
        "claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5-20251001",
        "claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus",
    )
    api_version = "2023-06-01"

    @property
    def default_base_url(self) -> str:
        return "https://api.anthropic.com"

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "Content-Type": "application/json",
        }

    def chat(self, messages, model, temperature, max_tokens, system_prompt=None,
              response_format=None) -> ChatResult:
        # Anthropic's API has no response_format/JSON-mode equivalent - the
        # caller's prompt-level "respond with only JSON" instruction (see
        # shared.llm_helpers.llm_json) is the only lever here, so this
        # parameter is accepted for interface consistency but not sent.
        model = self.resolve_model(model)
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            body["system"] = system_prompt
        resp = requests.post(f"{self.base_url}/v1/messages", headers=self._headers(),
                              json=body, timeout=DEFAULT_TIMEOUT)
        self._raise_for_status(self.name, resp)
        data = resp.json()
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        usage = data.get("usage", {})
        return ChatResult(
            text=text,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            finish_reason=data.get("stop_reason", "stop") or "stop",
            raw_model=data.get("model", model),
        )

    def stream(self, messages, model, temperature, max_tokens, system_prompt=None) -> Iterator[str]:
        model = self.resolve_model(model)
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if system_prompt:
            body["system"] = system_prompt
        with requests.post(f"{self.base_url}/v1/messages", headers=self._headers(),
                            json=body, timeout=DEFAULT_TIMEOUT, stream=True) as resp:
            self._raise_for_status(self.name, resp)
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "content_block_delta":
                    text = event.get("delta", {}).get("text")
                    if text:
                        yield text


class GeminiProvider(BaseLLMProvider):
    """Optional fourth provider - Gemini's REST API differs from the other three
    (API key as a query param, "contents" instead of "messages")."""

    name = "gemini"
    default_model = "gemini-2.0-flash"
    known_models = ("gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash")

    @property
    def default_base_url(self) -> str:
        return "https://generativelanguage.googleapis.com/v1beta"

    def _headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}

    @staticmethod
    def _to_contents(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        contents = []
        for m in messages:
            role = "model" if m.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})
        return contents

    def chat(self, messages, model, temperature, max_tokens, system_prompt=None,
              response_format=None) -> ChatResult:
        model = self.resolve_model(model)
        generation_config: Dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if response_format and response_format.get("type") == "json_object":
            generation_config["responseMimeType"] = "application/json"
        body: Dict[str, Any] = {
            "contents": self._to_contents(messages),
            "generationConfig": generation_config,
        }
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
        resp = requests.post(url, headers=self._headers(), json=body, timeout=DEFAULT_TIMEOUT)
        self._raise_for_status(self.name, resp)
        data = resp.json()
        candidate = data["candidates"][0]
        text = "".join(p.get("text", "") for p in candidate.get("content", {}).get("parts", []))
        usage = data.get("usageMetadata", {})
        return ChatResult(
            text=text,
            prompt_tokens=usage.get("promptTokenCount", 0),
            completion_tokens=usage.get("candidatesTokenCount", 0),
            finish_reason=candidate.get("finishReason", "STOP"),
            raw_model=model,
        )

    def stream(self, messages, model, temperature, max_tokens, system_prompt=None) -> Iterator[str]:
        model = self.resolve_model(model)
        body: Dict[str, Any] = {
            "contents": self._to_contents(messages),
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        url = f"{self.base_url}/models/{model}:streamGenerateContent?alt=sse&key={self.api_key}"
        with requests.post(url, headers=self._headers(), json=body, timeout=DEFAULT_TIMEOUT, stream=True) as resp:
            self._raise_for_status(self.name, resp)
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                try:
                    chunk = json.loads(line[len("data:"):].strip())
                except json.JSONDecodeError:
                    continue
                for cand in chunk.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        text = part.get("text")
                        if text:
                            yield text


class NativeTransformerProvider(BaseLLMProvider):
    """Runs DAXRO 1.0 - DATORYX's own from-scratch transformer (see
    models/native/) - not a downloaded pretrained model, not a cloud API.
    The weights are trained in this repo, on this repo's own source +
    docs, via `python -m models.native.train` (models/native/train.py) or
    the automated `backend/scripts/auto_train.sh` / `.ps1` scripts.

    This is the genuinely "built and trained by us" option: no external
    model file to download, no API key, no network call. It trades that
    for being a small, char-level model (see models/native/README.md for
    exactly what it is and isn't) - it won't produce fluent conversation
    the way a multi-billion-parameter pretrained model or a cloud API
    would. It's real, working, from-scratch inference, at the scale a
    CPU-only training run in this repo can actually support.
    """

    name = "native"
    display_name = "DAXRO 1.0"
    default_model = "daxro-1.0"
    known_models = ("daxro-1.0", "datoryx-native")

    def __init__(self, checkpoint_dir: str = None, prefer_best: bool = True, **_ignored):
        checkpoint_dir = checkpoint_dir or os.path.join(
            os.path.dirname(__file__), "..", "native", "checkpoint"
        )
        # If a training run used --patience (see models/native/train.py),
        # it saves the best-val_loss checkpoint separately as model_best.pt
        # alongside the final (possibly overfit) model.pt. Prefer that one
        # by default - it generalizes better than whatever step training
        # happened to stop at.
        best_ckpt_path = os.path.join(checkpoint_dir, "model_best.pt")
        final_ckpt_path = os.path.join(checkpoint_dir, "model.pt")
        vocab_path = os.path.join(checkpoint_dir, "vocab.json")

        if prefer_best and os.path.isfile(best_ckpt_path):
            ckpt_path = best_ckpt_path
        else:
            ckpt_path = final_ckpt_path

        if not os.path.isfile(ckpt_path) or not os.path.isfile(vocab_path):
            raise ValueError(
                "native: no trained checkpoint found. Run "
                "`python -m models.native.train` first (see models/native/README.md)."
            )
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "native provider requires torch. Install with: pip install torch"
            ) from exc

        from ..native.model import DatoryxNativeGPT, NativeConfig
        from ..native.tokenizer import CharTokenizer

        self.api_key = None
        self.base_url = "native://in-process"
        self.models = [self.default_model]
        self.checkpoint_dir = checkpoint_dir

        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = NativeConfig(**ckpt["config"])
        self._model = DatoryxNativeGPT(cfg)
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()
        self._tok = CharTokenizer.load(vocab_path)
        self._trained_steps = ckpt.get("step", 0)
        self._torch = torch

    @property
    def default_base_url(self) -> str:
        return "native://in-process"

    def _prompt_text(self, messages, system_prompt):
        """Build the char-level model's input in the same "User: ...\\n
        Assistant:" turn format used in models/native/chat_corpus/ (see
        train.py --extra-text-dir) - this is what lets it respond to
        conversational input like "hello" instead of just continuing
        arbitrary code, and what gives generation a clear cue (a trailing
        "Assistant:") to continue as a reply rather than random text.

        Only the most recent user turn is used - the model's context
        window (block_size, 128 chars by default) is far too small for
        multi-turn history, and idx_cond in model.generate() only looks at
        the last block_size characters anyway, so anything earlier would
        just get silently truncated regardless.
        """
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break
        if not last_user and messages:
            last_user = messages[-1].get("content", "")
        return f"User: {last_user}\nAssistant:"

    def chat(self, messages, model, temperature, max_tokens, system_prompt=None,
              response_format=None) -> ChatResult:
        prompt = self._prompt_text(messages, system_prompt)
        idx = self._torch.tensor([self._tok.encode(prompt)], dtype=self._torch.long)
        n_new = min(max_tokens, 400)
        with self._torch.no_grad():
            out = self._model.generate(idx, max_new_tokens=n_new,
                                        temperature=max(temperature, 0.1), top_k=40)
        full_text = self._tok.decode(out[0].tolist())
        completion = full_text[len(prompt):]
        # The model has no explicit stop token, so left unchecked it will
        # keep generating past a real answer into a fabricated "User: ..."
        # continuation of its own. Cut at the first sign of that - keeps
        # the returned text to just this one reply, matching what a chat
        # caller actually wants back.
        for stop in ("\nUser:", "\nuser:", "User:"):
            stop_idx = completion.find(stop)
            if stop_idx != -1:
                completion = completion[:stop_idx]
        completion = completion.strip()
        return ChatResult(
            text=completion,
            prompt_tokens=len(prompt),
            completion_tokens=len(completion),
            finish_reason="length",
            raw_model=f"datoryx-native (trained {self._trained_steps} steps)",
        )

    def stream(self, messages, model, temperature, max_tokens, system_prompt=None) -> Iterator[str]:
        # Generation isn't incrementally streamable token-by-token in this
        # minimal implementation - produce the full completion, then yield
        # it back in small chunks so callers using the streaming API still
        # get a sequence of chunks rather than one giant blob.
        result = self.chat(messages, model, temperature, max_tokens, system_prompt)
        chunk_size = 8
        for i in range(0, len(result.text), chunk_size):
            yield result.text[i:i + chunk_size]


class DaxroFineTunedProvider(BaseLLMProvider):
    """DAXRO 1.0's newer sibling: a real pretrained model (Qwen2.5-3B-
    Instruct) fine-tuned with a small LoRA adapter (see
    models/daxro_finetuned/) trained on DATORYX-specific identity/style
    data. Unlike NativeTransformerProvider (models/native/) - a tiny
    transformer trained completely from scratch on this repo's own code -
    this one starts from a competent, already-pretrained base model, so it
    can genuinely hold a conversation, write real code, and answer general
    questions. The fine-tuning only teaches it to identify as "DAXRO" and
    matches DATORYX's house style; it did not teach it to code or reason -
    the base model already could.

    Requires a CUDA GPU. bitsandbytes 4-bit quantization (how this was
    trained and how it's loaded here) needs CUDA - it will not load on a
    CPU-only host. If you're deploying on a CPU-only box (e.g. a typical
    free-tier Render/Vercel instance), this provider intentionally fails
    to register (see manager.py's _try_register_daxro_finetuned) rather
    than attempting a multi-GB fp32 CPU load that would be far too slow
    and memory-heavy to serve real requests. Use a GPU-backed host to run
    this in production; cloud providers or the tiny native model remain
    the CPU-friendly options.
    """

    name = "daxro"
    display_name = "DAXRO (fine-tuned)"
    default_model = "daxro-ft-1.0"
    known_models = ("daxro-ft-1.0",)

    def __init__(self, adapter_dir: str = None, base_model: str = None, **_ignored):
        adapter_dir = adapter_dir or os.path.join(
            os.path.dirname(__file__), "..", "daxro_finetuned", "adapter"
        )
        adapter_weights = os.path.join(adapter_dir, "adapter_model.safetensors")
        adapter_config = os.path.join(adapter_dir, "adapter_config.json")
        if not os.path.isfile(adapter_weights) or not os.path.isfile(adapter_config):
            raise ValueError(
                "daxro (fine-tuned): no LoRA adapter found in "
                f"{adapter_dir}. Train one via the Colab fine-tuning "
                "notebook and drop adapter_model.safetensors + "
                "adapter_config.json + tokenizer files there."
            )
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "daxro (fine-tuned) provider requires torch. Install with: pip install torch"
            ) from exc
        if not torch.cuda.is_available():
            # See the class docstring: bitsandbytes 4-bit needs CUDA, and a
            # 3B-parameter model in fp32 on CPU is impractical to serve.
            # Fail loudly and specifically so this doesn't silently become
            # a 10-second-per-token provider in production.
            raise RuntimeError(
                "daxro (fine-tuned) provider requires a CUDA GPU (needed for "
                "4-bit inference of the 3B-parameter base model). No GPU was "
                "detected - skipping registration. Deploy on a GPU-backed "
                "host to use this provider, or rely on cloud providers / the "
                "native char-level model on CPU-only hosts."
            )

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "daxro (fine-tuned) provider requires transformers + peft + "
                "bitsandbytes + accelerate. Install with: "
                "pip install transformers peft bitsandbytes accelerate"
            ) from exc

        with open(adapter_config) as f:
            adapter_cfg = json.load(f)
        base_model = base_model or adapter_cfg.get(
            "base_model_name_or_path", "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
        )

        self.api_key = None
        self.base_url = "daxro-finetuned://in-process"
        self.models = [self.default_model]
        self.adapter_dir = adapter_dir

        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb_config, device_map="auto",
        )
        self._model = PeftModel.from_pretrained(base, adapter_dir)
        self._model.eval()
        self._tok = AutoTokenizer.from_pretrained(adapter_dir)
        self._torch = torch
        self._base_model_name = base_model

    @property
    def default_base_url(self) -> str:
        return "daxro-finetuned://in-process"

    def chat(self, messages, model, temperature, max_tokens, system_prompt=None,
              response_format=None) -> ChatResult:
        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.extend(messages)
        inputs = self._tok.apply_chat_template(
            chat_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        ).to(self._model.device)
        with self._torch.no_grad():
            out = self._model.generate(
                input_ids=inputs,
                max_new_tokens=min(max_tokens, 800),
                temperature=max(temperature, 0.1),
                do_sample=True,
                pad_token_id=self._tok.eos_token_id,
            )
        completion = self._tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
        return ChatResult(
            text=completion,
            prompt_tokens=int(inputs.shape[-1]),
            completion_tokens=len(out[0]) - int(inputs.shape[-1]),
            finish_reason="stop",
            raw_model=f"daxro-ft-1.0 (base: {self._base_model_name})",
        )

    def stream(self, messages, model, temperature, max_tokens, system_prompt=None) -> Iterator[str]:
        # Same non-incremental streaming shim as NativeTransformerProvider -
        # generate the full reply, then chunk it for callers using the
        # streaming API.
        result = self.chat(messages, model, temperature, max_tokens, system_prompt)
        chunk_size = 8
        for i in range(0, len(result.text), chunk_size):
            yield result.text[i:i + chunk_size]


class DaxroRemoteProvider(BaseLLMProvider):
    """DAXRO fine-tuned model served remotely (e.g. on Modal with a GPU) as
    an OpenAI-shaped /chat/completions endpoint - see
    daxro_modal_deploy/app.py for the server side.

    This is the provider that actually gets used in production: the main
    DATORYX backend typically runs on a CPU-only host (Render/Vercel),
    which cannot run DaxroFineTunedProvider (in-process, requires a local
    CUDA GPU - see that class's docstring). DaxroRemoteProvider instead
    makes a plain HTTP call to wherever the GPU-backed model is actually
    hosted, so the main app needs no heavy ML dependencies (no torch,
    transformers, peft, bitsandbytes) at all for this to work.
    """

    name = "daxro-remote"
    display_name = "DAXRO (fine-tuned, remote)"
    default_model = "daxro-ft-1.0"
    known_models = ("daxro-ft-1.0",)

    def __init__(self, url: str, token: str = None):
        if not url:
            raise ValueError("daxro-remote: DAXRO_REMOTE_URL is required")
        self.api_key = token  # optional - only needed if the endpoint checks a shared secret
        self.base_url = url.rstrip("/")
        self.models = [self.default_model]

    @property
    def default_base_url(self) -> str:
        return self.base_url

    def chat(self, messages, model, temperature, max_tokens, system_prompt=None,
              response_format=None) -> ChatResult:
        payload = list(messages)
        if system_prompt and not any(m.get("role") == "system" for m in payload):
            payload = [{"role": "system", "content": system_prompt}] + payload
        body = {"messages": payload, "temperature": temperature, "max_tokens": max_tokens}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(self.base_url, headers=headers, json=body, timeout=DEFAULT_TIMEOUT)
        self._raise_for_status(self.name, resp)
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return ChatResult(
            text=choice["message"]["content"] or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
            raw_model=data.get("model", self.default_model),
        )

    def stream(self, messages, model, temperature, max_tokens, system_prompt=None) -> Iterator[str]:
        # daxro_modal_deploy/app.py doesn't implement SSE streaming (kept
        # simple - one endpoint, one JSON response) - shim it the same way
        # NativeTransformerProvider does for callers expecting a stream.
        result = self.chat(messages, model, temperature, max_tokens, system_prompt)
        chunk_size = 8
        for i in range(0, len(result.text), chunk_size):
            yield result.text[i:i + chunk_size]


class LocalLlamaProvider(BaseLLMProvider):
    """Runs a GGUF model in-process via llama-cpp-python.

    No API key, no network call, no external server - the model weights are
    loaded straight into this process. This is what makes DATORYX able to
    answer without any cloud provider configured. Quality/speed depend
    entirely on the model file you point it at (see
    scripts/setup_local_model.py) and your CPU/GPU.
    """

    name = "local"
    default_model = "local-gguf"
    known_models = ("local-gguf",)

    def __init__(self, model_path: str = None, n_ctx: int = None, n_threads: int = None,
                 n_gpu_layers: int = None, **_ignored):
        model_path = model_path or os.environ.get("DATORYX_LOCAL_MODEL_PATH")
        if not model_path or not os.path.isfile(model_path):
            raise ValueError(
                "local: no model file found. Set DATORYX_LOCAL_MODEL_PATH to a .gguf "
                "file, or run `python scripts/setup_local_model.py` to download one."
            )
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "local provider requires llama-cpp-python. Install with: "
                "pip install llama-cpp-python"
            ) from exc

        self.api_key = None
        self.base_url = "local://in-process"
        self.models = [self.default_model]
        self.model_path = model_path
        self.n_ctx = n_ctx or _env_int("DATORYX_LOCAL_MODEL_CTX", 4096)
        resolved_n_threads = n_threads or _env_int("DATORYX_LOCAL_MODEL_THREADS", os.cpu_count() or 4)
        resolved_n_gpu_layers = (
            n_gpu_layers if n_gpu_layers is not None
            else _env_int("DATORYX_LOCAL_MODEL_GPU_LAYERS", 0)
        )

        # CPU perf tuning - all overridable via env, all with safe defaults
        # for a plain laptop with no GPU:
        #
        # - n_batch: how many prompt tokens llama.cpp evaluates per batch
        #   during prompt processing. Higher batches process the prompt
        #   faster (fewer round-trips through the compute graph) at the
        #   cost of more RAM per batch; 512 is llama.cpp's own tuned
        #   default and works well on CPU. Lower it (e.g. 128-256) only on
        #   very RAM-constrained machines.
        # - n_threads_batch: threads used specifically for prompt
        #   processing, separate from n_threads (used for token-by-token
        #   generation). Prompt processing parallelizes better than
        #   generation, so this can safely equal the full core count even
        #   when n_threads is tuned down to leave a core free for the API
        #   server itself.
        # - use_mmap: memory-maps the GGUF file instead of reading it fully
        #   into the process's own heap - much faster process startup/model
        #   load and lets the OS page cache share pages across restarts.
        #   Safe to leave on for every deployment.
        # - use_mlock: locks the model's pages in physical RAM so the OS
        #   can never swap them out. Prevents a nasty worst case (a
        #   context-switch to disk mid-generation grinding CPU inference
        #   to a halt), but pins the whole model's RAM permanently and can
        #   fail without elevated privileges/ulimits on some systems -
        #   opt-in, off by default.
        # - flash_attn: llama.cpp's fused attention kernel. Meaningful
        #   speedup on both GPU and modern CPUs with AVX2; guarded so an
        #   older llama-cpp-python build that doesn't accept the kwarg
        #   degrades gracefully instead of crashing the whole app.
        n_batch = _env_int("DATORYX_LOCAL_MODEL_BATCH", 512)
        n_threads_batch = _env_int("DATORYX_LOCAL_MODEL_THREADS_BATCH", resolved_n_threads)
        use_mmap = os.environ.get("DATORYX_LOCAL_MODEL_MMAP", "true").strip().lower() not in ("0", "false", "no")
        use_mlock = os.environ.get("DATORYX_LOCAL_MODEL_MLOCK", "false").strip().lower() in ("1", "true", "yes")
        flash_attn = os.environ.get("DATORYX_LOCAL_MODEL_FLASH_ATTN", "true").strip().lower() not in ("0", "false", "no")

        base_kwargs = dict(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_threads=resolved_n_threads,
            n_gpu_layers=resolved_n_gpu_layers,
            n_batch=n_batch,
            n_threads_batch=n_threads_batch,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            verbose=False,
        )
        self._llm = self._instantiate(Llama, base_kwargs, flash_attn=flash_attn)

    @staticmethod
    def _instantiate(llama_cls, base_kwargs: dict, flash_attn: bool):
        """Build the Llama() instance, degrading gracefully on older
        llama-cpp-python builds that don't recognize a given constructor
        kwarg yet, instead of hard-crashing the whole process on startup
        over a performance knob. Tries progressively simpler kwarg sets."""
        attempts = []
        if flash_attn:
            attempts.append({**base_kwargs, "flash_attn": True})
        attempts.append(dict(base_kwargs))
        # Last-resort minimal set, in case an even older build also
        # rejects n_threads_batch/use_mlock/use_mmap.
        minimal = {k: v for k, v in base_kwargs.items()
                   if k in ("model_path", "n_ctx", "n_threads", "n_gpu_layers", "verbose")}
        attempts.append(minimal)

        last_exc = None
        for kwargs in attempts:
            try:
                return llama_cls(**kwargs)
            except TypeError as exc:
                last_exc = exc
                continue
        raise last_exc

    @property
    def default_base_url(self) -> str:
        return "local://in-process"

    def _build_messages(self, messages, system_prompt):
        payload = list(messages)
        if system_prompt and not any(m.get("role") == "system" for m in payload):
            payload = [{"role": "system", "content": system_prompt}] + payload
        return payload

    def _fit_to_context(self, messages, max_tokens):
        """Drop oldest non-system turns if the conversation is too long for
        this model's context window, instead of letting llama.cpp error out.
        Rough token estimate (chars/4) - conservative on purpose, since
        going slightly over-cautious just means trimming one extra turn,
        while under-estimating means a hard crash mid-request."""
        budget = self.n_ctx - max_tokens - 128  # headroom for chat template overhead
        if budget <= 0:
            budget = max_tokens  # degenerate config; still try rather than refuse outright

        def _tokens(msgs):
            return sum(len(m.get("content", "")) for m in msgs) // 4

        system = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]
        while len(rest) > 1 and _tokens(system + rest) > budget:
            rest.pop(0)  # drop oldest turn first, but always keep the latest one
        return system + rest

    def chat(self, messages, model, temperature, max_tokens, system_prompt=None,
              response_format=None) -> ChatResult:
        payload = self._fit_to_context(self._build_messages(messages, system_prompt), max_tokens)
        kwargs = {}
        if response_format and response_format.get("type") == "json_object":
            # llama-cpp-python enforces a JSON grammar at the sampling level
            # for this - not just a prompt instruction the model might
            # ignore. This is the actual reliability win for a small local
            # model doing structured decisions: syntactically-invalid JSON
            # becomes impossible to emit, not just "less likely".
            kwargs["response_format"] = {"type": "json_object"}
        try:
            out = self._llm.create_chat_completion(
                messages=payload,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as exc:
            # Normalize any llama.cpp failure (OOM, malformed input, context
            # overflow we didn't catch, etc.) into the same error type cloud
            # providers raise, so LLMManager's existing retry/fallback logic
            # (try next provider, then degrade to simulated) applies here too
            # instead of the request crashing outright.
            raise LLMProviderError("local", str(exc), retryable=False) from exc
        choice = out["choices"][0]
        usage = out.get("usage", {})
        return ChatResult(
            text=choice["message"]["content"] or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
            raw_model=self.model_path,
        )

    def stream(self, messages, model, temperature, max_tokens, system_prompt=None) -> Iterator[str]:
        payload = self._fit_to_context(self._build_messages(messages, system_prompt), max_tokens)
        try:
            out = self._llm.create_chat_completion(
                messages=payload,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in out:
                delta = chunk["choices"][0].get("delta", {})
                text = delta.get("content")
                if text:
                    yield text
        except Exception as exc:
            raise LLMProviderError("local", str(exc), retryable=False) from exc


#: name -> adapter class, used by LLMManager for both auto-registration
#: (env var scan) and explicit register_provider() calls.
PROVIDER_REGISTRY = {
    "groq": GroqProvider,
    "grok": GrokProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "local": LocalLlamaProvider,
    "daxro-remote": DaxroRemoteProvider,
    "daxro": DaxroFineTunedProvider,
    "native": NativeTransformerProvider,
}

#: provider -> env var holding its API key, used for auto-registration
PROVIDER_ENV_KEYS = {
    "groq": "GROQ_API_KEY",
    "grok": "GROK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
