# DATORYX 

A local, MCP-native multi-agent AI operator — built by Anucodex21
[DATORYX](.) (21 specialized LLM-backed agents, brain/memory layers) into a
single system any MCP client, browser, or REST client can drive.

**New here?** See [`START_HERE.md`](START_HERE.md) for the fully
walkthrough — API keys, running locally, training DAXRO 1.0, and
deploying to GitHub + Render + Vercel, all in order. This README covers
architecture and what's been tested.

## What this is

DATORYX already had the hard parts: 20 real, LLM-backed specialized agents
(coding, research, data science, security, devops, sql, forecasting, ...),
a shared multi-provider `LLMManager` with local/offline fallback, and a
memory system (episodic / semantic / vector / long-term / short-term /
knowledge graph). What it didn't have was a standard way for outside tools
to talk to it.

This repo adds that layer: **`backend/datoryx_mcp/mcp_server.py`** walks
every agent at startup and auto-generates one MCP tool per capability —
**120 tools across 20 agents**, entirely generated from the existing
`AgentCapability` metadata, with zero manual per-tool wiring. Add a new
agent under `backend/agents/<name>/agent.py` following the existing
`BaseAgent` contract and it appears as MCP tools automatically next run.

## Why this over a plain "MCP client" AI stack

Most local AI-operator projects (this one included, before today) are MCP
**clients**: they connect out to other people's MCP servers (filesystem,
GitHub, browser, security scanners) and call those tools. That's useful,
but it means the AI agents themselves aren't reusable by anything else.

This repo makes the agent fleet an MCP **server** too:

- Any MCP client — Claude Desktop, Claude Code, Cursor, a custom UI —
  can connect to `datoryx-operator` and get all 20 agents as tools.
- The system can *also* still act as an MCP client (see
  `backend/datoryx_mcp/` — add client-side connections here) so agents can
  call out to filesystem/browser/security MCP servers when a task needs it.
- Net result: bidirectional MCP, not just one direction.

## Quick start — as a product (browser + REST API)

The whole point of this layer: no MCP client required. Register an
account, open the dashboard, and drive all 20 agents from a browser.

```bash
docker compose up --build
```

Then open **http://localhost:8000** — dark instrument-panel dashboard,
register/login (multi-user JWT auth, DATAGRID-style), pick an agent from
the roster, run a capability or ask it a free-form question.

Without Docker:

```bash
cd backend
pip install -r requirements.txt fastapi "uvicorn[standard]" bcrypt pyjwt email-validator
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Set provider keys in `.env` (copy `.env.example`) to get real LLM output
instead of the labeled offline-simulated response — the app runs and every
endpoint works with zero keys set, so you can verify the whole product
before spending any API credit.

**LLM routing / fallback chain** (highest priority first):

1. **`local`** — a GGUF model file on disk (llama.cpp), if configured
2. **`native` (DAXRO 1.0)** — DATORYX's own from-scratch trained model
   (`backend/models/native/checkpoint/`), if a checkpoint exists. This repo
   ships one already trained to 501,260 steps (trained on Colab's free T4
   GPU). **Note on that checkpoint:** validation loss was actually lowest
   around step 5,400 (val_loss 0.82) and rose steadily after that even as
   training loss kept falling (val_loss 1.8+ by step 501,260) — classic
   overfitting. It still runs and generates real output, but don't expect
   coherent general text from it; it's closer to a compressed, somewhat
   garbled memory of its own training data (DATORYX's source + docs) than
   a fluent model. Good enough as a working offline fallback and a real
   from-scratch-training learning exercise; not a substitute for a cloud
   provider's output quality. Train your own with `scripts/auto_train.sh` /
   `scripts/setup_cron.sh` — stopping earlier (~5,000-10,000 steps) will
   generalize better than training longer did here. **This has now been
   fixed at the training-script level** — `models/native/train.py` accepts
   `--patience N`: it auto-saves the best-val_loss checkpoint separately as
   `model_best.pt` on every improvement, and stops training early if
   val_loss hasn't improved in N evals. `NativeTransformerProvider` prefers
   `model_best.pt` over `model.pt` automatically when both exist. Retrain
   with `bash scripts/auto_train.sh 50000` after adding `--patience 20` to
   the underlying `train.py` call (or run `train.py` directly with
   `--patience 20`) to get a properly early-stopped checkpoint instead of
   the current overfit one.
3. **`groq`** → **`openai`** → **`anthropic`** → **`gemini`** — whichever
   cloud keys you set in `.env`, tried in that order on failure.

You don't need all of them — set whichever keys you have or none at all.
One note: a **Groq key alone already gets you Llama access**, since Groq
hosts Meta's Llama 3.x models — there's no separate "Llama API" to sign up
for unless you specifically want to run a Llama GGUF file locally via the
`local` provider instead of Groq's hosted one.

REST API surface (also browsable at `/docs`):

| Endpoint | What it does |
|---|---|
| `POST /auth/register`, `POST /auth/login` | Multi-user JWT auth |
| `GET /agents` | List all 20 agents + their capabilities |
| `POST /agents/{key}/execute` | Run a specific structured capability |
| `POST /chat` | Ask any agent a free-form question |

## Quick start — as an MCP server (for AI clients)

```bash
cd backend
pip install -r requirements.txt mcp
python -m datoryx_mcp.mcp_server
```

Then point any MCP client at it, e.g. Claude Desktop's
`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "datoryx": {
      "command": "python",
      "args": ["-m", "datoryx_mcp.mcp_server"],
      "cwd": "/absolute/path/to/datoryx-operator/backend"
    }
  }
}
```

Both entry points (`api/main.py` and `datoryx_mcp/mcp_server.py`) sit on
top of the same `AgentRegistry` — one codebase, two ways in.

## Layout

```
backend/
  agents/            21 specialized agents (20 from DATORYX + market_analyst) + registry.py
  market_data/       real pandas indicator math (from AI-Market-Analyzer) backing market_analyst
  models/llm/        LLMManager - multi-provider routing, offline fallback
  models/native/     DATORYX's own from-scratch trainable model
  memory/            session_store.py - real, tested per-user/per-agent chat memory (SQLite)
                     vector/ episodic/ etc. - DATORYX's fuller memory subsystem, not yet wired in (needs chromadb)
  shared/            shared types + llm_helpers (JSON-safe LLM calls)
  core/plugin_manager/  plugin loading + mcp_client.py (agents calling OUT to external MCP servers)
  datoryx_mcp/       MCP server: auto-exposes every agent as MCP tools (server direction)
  api/               Product API: FastAPI + multi-user JWT auth (browser/product entry point)
frontend/
  index.html         Dark instrument-panel dashboard - single file, no build step
mcp_servers.example.json   config template for external MCP servers agents can call out to
Dockerfile, docker-compose.yml   one-command deploy
```

## Tested end to end

- `POST /auth/register` → `200`, issues JWT
- `GET /agents` without a token → `401` (auth actually enforced)
- `GET /agents` with a token → `200`, **21 agents** returned
- `POST /agents/coding/execute` (`write_code`) → `200`, real `AgentResult`
- `POST /agents/market_analyst/execute` (`analyze_ohlcv`) → `200`, **real pandas-computed indicators** (RSI, MACD, SMA, ADX) synthesized by the LLM into a readable summary
- `POST /chat` (free-form to `research` agent), two turns → `200` both times, **second turn's prompt includes the first turn's history** (memory/session_store.py), verified isolated per-user
- `POST /auth/login` with wrong password → `401`
- `_call_external_tool()` on an unconfigured MCP server → returns a clean error dict, doesn't crash (degrade-don't-crash, verified)
- MCP server startup → **124 tools registered across 21 agents**, no crash even with zero provider API keys configured

## What's real vs. what's still a stub

Being upfront about the current state:

- **Real and tested**: MCP server generation, product API + auth, market_analyst's indicator math, session memory, external-tool call plumbing (degrades cleanly).
- **Wired but needs your config**: LLM output quality - without a provider API key in `.env`, every agent still runs and returns a labeled simulated response instead of crashing, but you won't get real generated code/analysis/text until a key is set.
- **Plumbing exists, needs a running external MCP server to actually do something**: `_call_external_tool()` / `mcp_servers.json` - copy in a real MCP server config (e.g. `@modelcontextprotocol/server-filesystem`) and it'll work; with nothing configured it correctly no-ops.
- **Not yet wired into this build**: NEXUS-GPT local model serving, AI-Master's RAG/speech/vision routes, Jarvis's voice pipeline, DATORYX's heavier vector/episodic memory (needs chromadb + a DB layer not included here to keep the build light). These are still real, working code in the other zips you sent - they're just not plugged into *this* unified registry yet. Next roadmap items if you want them in.


## Roadmap (next milestones)

1. ~~**MCP server**~~ — done: 124 tools, 21 agents, tested.
2. ~~**MCP client side**~~ — done: `core/plugin_manager/mcp_client.py` + `BaseAgent._call_external_tool()`, tested to degrade cleanly with no config.
3. ~~**market_analyst agent**~~ — done: real indicator math from AI-Market-Analyzer, 21st agent.
4. ~~**Session memory**~~ — done (lightweight SQLite version; DATORYX's fuller chromadb-backed vector/episodic memory is a heavier future upgrade).
5. **NEXUS-GPT local serving** — wire in the dynamic-batching local model server as another `LLMManager` provider, so the operator can run fully offline on consumer hardware.
6. **AI-Master RAG/speech/vision routes** — expose as additional MCP tools the same way agent capabilities are, via the registry pattern.
7. **Jarvis voice pipeline** — wake-word + TTS as an alternate interface into the same `/chat` endpoint.
8. ~~**DATORYX native model training**~~ — done: `scripts/auto_train.sh` (resumable manual training) + `scripts/nightly_train.sh` + `scripts/setup_cron.sh` (fully automated nightly training via cron, tested).
9. **Deploy** — Vercel (frontend static) + Render/VPS (backend, same pattern as the original DATORYX deploy).
10. **Public README + demo** — once deployed, a polished GitHub presentation to match the substance.
