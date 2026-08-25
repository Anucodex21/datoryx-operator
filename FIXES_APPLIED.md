# What was fixed in this build

## 1. Provider priority bug (`backend/models/llm/manager.py`)

**Before:** if `backend/models/native/checkpoint/model.pt` existed, DAXRO 1.0
(the native trained model) silently became the *default* provider — even
if you had `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` set. This happened
because native registered itself before cloud keys were even read, and
grabbed the default slot on the spot.

**After:** registration order changed (cloud keys register first) and a
new `_finalize_default_provider()` step runs once everything is known:

1. An explicit `llm.default_provider` config value, if registered.
2. Any cloud provider with a real API key set — **this now always wins**
   over local/native.
3. `local` (a GGUF file), if present.
4. `native` (DAXRO 1.0), if present — used only when no cloud key and no
   local model are configured.

DAXRO 1.0 is still fully registered and usable — as an automatic fallback
if every cloud provider call fails, and explicitly any time via
`provider="native"` in a call — it just no longer *silently* takes over
as the default the moment its checkpoint file exists.

Also fixed `LLMManager.mode()`, which previously reported `"native"`
whenever a checkpoint was present regardless of what was actually
answering calls. It now reports whichever provider is actually the
default.

Verified with a mocked-provider test (no `torch`/network required):
cloud key + native checkpoint both present → default is the cloud
provider, `mode()` reports `"cloud"`; no cloud key, checkpoint present →
default is `native`, `mode()` reports `"native"`.

## 2. Best DAXRO 1.0 checkpoint wired in

`backend/models/native/checkpoint/` now contains the checkpoint from your
best run (9x-larger, multi-project corpus, patience-based early stopping):

- `model.pt` — step 23,600, val_loss **0.8315** (auto-stopped by patience
  before it could overfit further; run continued to step 29,600 but never
  beat this val_loss)
- `vocab.json` — the matching 138-token vocab (includes the emoji set from
  this run's corpus)
- `train_log.json` — full loss curve for this run

## 3. `.env.example` updated

Documents the corrected priority order and clarifies that setting a cloud
key does not disable DAXRO 1.0 — it's still reachable, just not the
default anymore.

---

## To deploy with cloud APIs active (recommended)

Set at least one of these in your Render/Vercel environment variables:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

Agents will use that provider by default. DAXRO 1.0 stays available as a
fallback and via explicit `provider="native"` calls — it just won't be
the one answering by default anymore.

## To keep improving DAXRO 1.0 separately

Nothing about training changes — keep using
`python -m models.native.train --steps ... --patience ... --extra-text-dir ...`
as before. Each new best checkpoint just needs `model.pt` + `vocab.json`
(and ideally `train_log.json`) dropped into
`backend/models/native/checkpoint/` to take effect — cloud APIs will still
take priority automatically once a key is set, so there's no conflict
between continuing to train it and running the app on Claude/OpenAI.

## 4. Grok (xAI) added as a new provider

`GROK_API_KEY` is now a first-class provider alongside Groq — they are
different companies/APIs with similar names:

- **Groq** — fast inference host for Llama/Mixtral/Gemma models (`GROQ_API_KEY`)
- **Grok** — xAI's own model (`GROK_API_KEY`, models: `grok-4`, `grok-4-fast`, `grok-3`, `grok-3-mini`)

Both are OpenAI-wire-compatible, so `GrokProvider` reuses the same
`OpenAICompatibleProvider` base class as Groq/OpenAI — just pointed at
`https://api.x.ai/v1`. Set `GROK_API_KEY` in your env and it auto-registers
exactly like every other provider; call it explicitly with `provider="grok"`
or `model="grok-4"`.

## 5. Native model (DAXRO 1.0) now has a chat format

**Problem:** the checkpoint you have was trained only on source code +
docs. It had no "User: ... / Assistant: ..." pattern anywhere, and the
inference code just concatenated raw text with no structure - so asking
it "hello" produced random code-like continuation, not a reply.

**Fixed two things:**

1. `backend/models/native/chat_corpus/conversations.txt` — a new ~211k
   character corpus of `User: ... / Assistant: ...` exchanges (greetings,
   "who are you", small talk, basic factual Q&A), repeated with reshuffled
   ordering so a tiny model gets enough exposure to actually learn the
   pattern instead of it being lost in the much larger code corpus.

2. `backend/models/llm/providers.py` (`NativeTransformerProvider`):
   - `_prompt_text()` now builds `"User: {message}\nAssistant:"` instead
     of dumping raw concatenated text - this gives the model a consistent
     cue matching the new training format, so generation naturally
     continues as a reply.
   - `chat()` now truncates output at the first `"User:"` it generates,
     since the model has no real stop token and would otherwise keep
     rambling into a fabricated follow-up exchange.

Both changes were unit-tested (pure Python, no torch needed) — prompt
building and stop-truncation both verified correct.

**You still need to retrain** for the chat corpus to actually take
effect — code changes alone don't change what the existing checkpoint
learned. Run (same Colab flow as before):

```
!cd /content/backend && python -m models.native.train --steps 40000 --patience 25 \
    --extra-text-dir /content/backend/models/native/chat_corpus \
    --out-dir /content/backend/models/native/checkpoint
```

(If you're also including your other projects' code via a separate
`--extra-text-dir`, merge `chat_corpus/conversations.txt` into that same
folder first — `train.py` only accepts one `--extra-text-dir`.)

**What to expect realistically:** with a 4-layer, 128-dim model and
128-char context, it will learn the covered patterns (hello/hi, "who are
you", the Q&A pairs included) reasonably well since they're heavily
repeated. It will NOT handle genuinely novel questions outside that
set with real understanding - that's a fundamental size limit, not a bug.
For open-ended chat, that's what the cloud providers (Groq/Grok/OpenAI/
Anthropic) are for; DAXRO 1.0 is the always-available, zero-cost
fallback for simple, in-distribution exchanges.

## 6. Fixed a train/val leakage risk in the chat corpus

The first version of `chat_corpus/conversations.txt` repeated the same
~260 exchanges 8x with reshuffled block order to boost its weight against
the much larger code corpus. Problem: `train.py`'s train/val split is a
simple positional 90/10 slice (`data[:n]`, `data[n:]`), not shuffled - so
if those literal repeats straddled the split point, near-identical
content could land in both train and val, making val_loss look
artificially good (and confusing patience-based early stopping) without
reflecting real generalization.

Regenerated with genuine paraphrase variety instead of duplication —
585 exchanges, **zero exact-duplicate lines**, 68,353 chars total. Same
idea (heavy coverage of greetings/small-talk/basic Q&A) but every
occurrence is textually distinct, so there's no leakage risk regardless
of where the 90/10 split lands.

## 7. Chat-trained checkpoint integrated (2nd training run)

Swapped in the checkpoint from the chat-corpus retraining run:

- Trained on repo corpus (253,239 chars) + chat_corpus (68,353 chars) = 321,594 chars total
- Auto-stopped via `--patience 25` at step 22,400
- Best val_loss: **0.1479** at step 17,400 (`model_best.pt`)
- Verified structurally consistent: checkpoint's internal `config.vocab_size`
  (96) matches the paired `vocab.json` (96 tokens) exactly, and the internal
  `step` (17400) matches the best val_loss entry in `train_log.json`

Files replaced in `backend/models/native/checkpoint/`: `model.pt`,
`model_best.pt`, `vocab.json`, `train_log.json`.

**Not yet independently verified**: actual chat output quality (e.g. what
it says for "hello"). Structural checks (vocab size, config, step) all
match correctly, but confirming the checkpoint loads and produces sensible
replies needs `torch`, which isn't available in the environment used to
prepare this package. Run the test snippet provided earlier (loads
`model_best.pt` and prints replies for "hello", "who are you", etc.) before
relying on this in production, and let me know what it outputs if anything
looks off.

## 8. DAXRO fine-tuned model integrated (Path B - real pretrained + LoRA)

`backend/models/daxro_finetuned/adapter/` now contains the LoRA adapter
trained on top of `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` (34-example
instruction dataset: identity, coding, general knowledge, DATORYX project
info).

**New provider**: `DaxroFineTunedProvider` in `models/llm/providers.py`,
registered as `"daxro"` in `models/llm/manager.py`. Full priority chain,
verified with tests:

```
cloud (Groq/Grok/OpenAI/Anthropic/Gemini, if a key is set)
  > local (llama.cpp GGUF, if configured)
    > daxro (this fine-tuned model, if a GPU + the adapter are present)
      > native (the original from-scratch char-level model)
```

**Critical requirement - GPU only.** This provider requires CUDA (the
4-bit quantized base model won't load on CPU). It checks
`torch.cuda.is_available()` at registration time and **intentionally
fails to register** (with a clear warning, not a crash) if no GPU is
present - so it silently steps aside on CPU-only hosts like a typical
free-tier Render/Vercel deployment, and the chain falls through to cloud
providers or the native model instead. To actually use it in production,
deploy on a GPU-backed host and install the optional dependencies now
listed (commented out) in `requirements.txt`: `transformers`, `peft`,
`bitsandbytes`, `accelerate` (plus `torch`, already needed for `native`).

**Structurally verified** (adapter_config.json inspected): base model
`unsloth/Qwen2.5-3B-Instruct-bnb-4bit`, `r=16`, `lora_alpha=16`, target
modules match what the Colab training script specified - consistent
with a successful training run.

**Not yet verified**: actual generation quality/output, since this
sandbox has no GPU. Test it yourself with the Cell 6 snippet from the
Colab fine-tuning steps (or call the `/chat` endpoint with
`provider="daxro"` once deployed on a GPU host) before demoing to your
company.

## 9. DAXRO fine-tuned - production path via Modal (remote GPU)

Since the main app runs on Render (CPU-only), `DaxroFineTunedProvider`
(in-process) can't actually load there. Added the real production path:

- **`daxro_modal_deploy/`** — a standalone Modal app (`app.py`) that serves
  the same Qwen2.5-3B + LoRA adapter as an OpenAI-shaped
  `/chat/completions` HTTP endpoint, on a T4 GPU. Deploy separately with
  `modal deploy app.py` (see `daxro_modal_deploy/README.md`).
- **`DaxroRemoteProvider`** (new, in `providers.py`) — calls that Modal
  endpoint over plain HTTP. No torch/transformers/peft needed in the main
  app for this to work.
- **`manager.py`** — registers `daxro-remote` when `DAXRO_REMOTE_URL` is
  set. Full priority chain (tested):

  ```
  cloud (Groq/Grok/OpenAI/Anthropic/Gemini, if a key is set)
    > local (llama.cpp GGUF, if configured)
      > daxro-remote (Modal-hosted DAXRO, if DAXRO_REMOTE_URL is set)
        > daxro (in-process DAXRO, only works if this host itself has a GPU)
          > native (original from-scratch char-level model)
  ```

- `render.yaml` and `.env.example` updated with `DAXRO_REMOTE_URL` /
  `DAXRO_REMOTE_TOKEN`.

**To actually go live with DAXRO fine-tuned**: follow
`daxro_modal_deploy/README.md`, then set `DAXRO_REMOTE_URL` in Render's
environment variables. Everything else (cloud APIs, native model) already
works without this - it's additive.

## 10. Extended features - honest, real implementations

Added the missing features from your screenshots (Claude.ai / ChatGPT /
Antigravity-style sidebars), each genuinely working within a clearly
stated scope - not faked, not just UI with no backend:

| Feature | What it actually does | Tested |
|---|---|---|
| **Swarm** | Dispatches a query to several relevant agents (auto-picked by keyword match, or specify agent keys) concurrently, combines results with the LLM | Logic-tested |
| **Plugins** | Exposes the existing MCP client (`core/plugin_manager/mcp_client.py`) - lists configured servers + their tools | N/A - depends on your mcp_servers.json |
| **Docs** | LLM drafts structured content -> real `.docx` via python-docx | **Verified**: generated file re-opened and checked |
| **Slides** | Same pattern -> real `.pptx` via python-pptx | **Verified**: generated file re-opened and checked |
| **Sheets** | Same pattern -> real `.xlsx` via openpyxl | **Verified**: generated file re-opened and checked |
| **Websites** | LLM drafts a self-contained HTML file, saved as a downloadable artifact. **Not** live-hosted/deployed - that's a separate, larger feature | **Verified**: generation + extraction logic tested |
| **Deep Research** | Breaks a topic into subtopics, researches each, synthesizes a report. **Reasoning-only by default** (labeled as such); becomes **web-grounded** automatically if `TAVILY_API_KEY` is set | **Verified**: orchestration tested end-to-end with a fake LLM |
| **Images** | Calls OpenAI's image API using your existing `OPENAI_API_KEY`. Returns a clear error (not a fake image) if that key isn't set | N/A - needs a real OpenAI key + network to test live |
| **Artifacts** | Lists/downloads everything generated by Docs/Slides/Sheets/Websites, per user | Logic-tested |
| **Projects** | Simple named groupings of artifacts (metadata only) | Logic-tested |
| **Customize** | Per-user custom instructions, saved/loaded | Logic-tested |
| **Scheduled Tasks** | Lightweight in-process scheduler (thread + loop, no new dependency). **Caveat surfaced in the API response itself**: only fires while the server process stays running - won't fire while a free-tier host is asleep from inactivity | Logic reviewed, not runtime-tested (needs a live server) |
| **Code** | Not a new feature - your existing "Coding" agent already covers this. Just select it from the agent roster (noted in the UI's empty state) | Pre-existing |

**Not built** (would need scope/cost decisions you haven't made yet):
- **Projects/Artifacts/Code/Customize as Claude.ai has them** (full workspace with file versioning, persistent conversation branching) - what's here is a genuinely working but much simpler version of the same idea.
- **Live website deploy** (one-click hosting) - "Websites" here produces a real downloadable HTML file only.
- **Image editing/variations, multiple images per request** - only single-image generation via OpenAI is wired up.

**New files**: `backend/generators/` (docs/slides/sheets/website/research
generation logic), `backend/api/extended_features.py` (all new endpoints).
**Frontend**: `frontend/index.html` got a new feature bar with a working
panel for each of the 12 features above, calling the real endpoints.

**Storage caveat** (same as the existing `users.json` pattern): artifacts,
projects, customize preferences, and tasks are stored as JSON files on
local disk. On a free-tier host with an ephemeral filesystem, this data
does **not** survive a redeploy. Fine for a demo; swap for a real
database before relying on it long-term - same caveat the existing
`api/auth.py` already documents for user accounts.

**Requirements added**: `python-docx`, `python-pptx`, `openpyxl` (all
already verified working in this environment). `TAVILY_API_KEY` is
optional (only upgrades Research to web-grounded mode).
