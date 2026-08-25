# START HERE — Everything in one place

This is the master checklist. Detailed reference docs also exist
(`README.md` for architecture/testing, `DEPLOY.md` for deploy command
reference, `backend/models/native/TRAINING_GUIDE.md` for training deep
dive) — this file is the order to actually do things in.

---

## 1. API keys — where and how

**Where:** a file named `.env` in the project root (`datoryx-operator/.env`).
It doesn't exist yet — copy the template:

```bash
cp .env.example .env
```

Open `.env` in any text editor. It looks like this:

```
DATORYX_JWT_SECRET=change-me-to-a-long-random-string
GROQ_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
```

Paste whichever key(s) you have after the `=` sign, e.g.:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

**You don't need all four.** One is enough to get real LLM output. Zero
is also fine — the app still runs, agents just return a labeled
simulated response instead of a real one.

**Where to get a free key fast:** [console.groq.com](https://console.groq.com) →
sign up → API Keys → Create. Groq is free-tier and also gives you Llama
3.x model access, so it's the easiest single key to start with.

**This same `.env` file is used for local runs.** For the deployed
version on Render, you paste the same keys into Render's dashboard
instead (step 4 below) — `.env` itself never gets uploaded anywhere
(it's git-ignored on purpose, so your keys don't end up on GitHub).

---

## 2. Run it locally on your laptop

```bash
cd datoryx-operator
docker compose up --build
```

Open **http://localhost:8000** in a browser. Register an account, log
in, pick an agent from the roster, run it. That's the whole product,
running entirely on your machine.

No Docker installed? Run it directly instead:
```bash
cd backend
pip install -r requirements.txt fastapi "uvicorn[standard]" bcrypt pyjwt email-validator
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

To also use it as an MCP server (so Claude Desktop / Claude Code can call
your 21 agents as tools):
```bash
cd backend
python -m datoryx_mcp.mcp_server
```

---

## 3. Training DAXRO 1.0 (optional — the app works without this)

A checkpoint already ships in `backend/models/native/checkpoint/` (trained
1,260 steps). You only need this section if you want it smarter.

**One-time setup:**
```bash
cd backend
pip install torch
```

**Option A — manual, whenever you feel like it:**
```bash
bash scripts/auto_train.sh 20000
```
Trains toward 20,000 total steps. Safe to stop (Ctrl+C) and re-run later —
always resumes from the checkpoint on disk, never restarts from zero.
Rough timing: ~0.5–1 second per step on a normal laptop CPU, so +5,000
steps ≈ 45–90 minutes. A free Colab/Kaggle GPU does the same step count
in a couple of minutes — see `backend/models/native/TRAINING_GUIDE.md`
for that walkthrough if you want it faster.

**Avoiding overfitting — use `--patience`:** the checkpoint that ships in
this repo was trained without early stopping and overfit badly (best
generalization was around step 5,400; it kept training to 500,000+ and
got worse from there). Run `train.py` directly instead of `auto_train.sh`
if you want this protection:
```bash
python -m models.native.train --steps 50000 --patience 20 --resume
```
This stops automatically once val_loss hasn't improved in 20 evals, and
separately saves the best-val_loss checkpoint as `model_best.pt` — which
`NativeTransformerProvider` prefers automatically over `model.pt` when
both exist, so you don't need to manually pick a step count.

**Option B — fully automatic, every night, hands-off:**
```bash
bash scripts/setup_cron.sh
```
One command, one time. From then on, every night at 2:30 AM it trains
+4,000 steps on its own, resuming safely if interrupted, capping out at
100,000 total steps. Your laptop needs to be on and awake at that time
for it to fire.

Pause it anytime without uninstalling the schedule:
```bash
bash scripts/pause_training.sh    # skip nights until you resume
bash scripts/resume_training.sh   # turn it back on
```

Watch it work: `tail -f backend/models/native/checkpoint/nightly_train.log`

---

## 4. Deploy — GitHub → Render (backend) → Vercel (frontend)

Do these in order; each step needs the previous one done first.

### 4a. Push to GitHub
```bash
cd datoryx-operator
gh repo create Anucodex21/datoryx-operator --public --source=. --remote=origin
git push -u origin main
```
No `gh` CLI? Create the repo manually at github.com/new (name:
`datoryx-operator`, don't add a README), then:
```bash
git remote add origin https://github.com/Anucodex21/datoryx-operator.git
git push -u origin main
```

### 4b. Deploy the backend — Render
1. [render.com](https://render.com) → **New** → **Blueprint**
2. Connect the `Anucodex21/datoryx-operator` repo — Render reads
   `render.yaml` automatically and configures the Docker service for you.
3. It'll prompt for env vars — paste in the same keys from your `.env`
   (`GROQ_API_KEY`, etc.). `DATORYX_JWT_SECRET` auto-generates itself.
4. Click deploy. You get a URL like:
   `https://datoryx-operator-api.onrender.com`
5. Confirm: `curl https://datoryx-operator-api.onrender.com/health`
   should return `{"status":"ok",...}`.

Free-tier Render sleeps after inactivity, ~30s to wake on first request —
normal.

### 4c. Deploy the frontend — Vercel
1. First, point the dashboard at your live backend. Edit
   `frontend/index.html`, find this line near the top of `<body>`:
   ```html
   window.DATORYX_API_BASE = "";
   ```
   Change it to your Render URL:
   ```html
   window.DATORYX_API_BASE = "https://datoryx-operator-api.onrender.com";
   ```
   Save, then:
   ```bash
   git add frontend/index.html
   git commit -m "Point dashboard at live Render backend"
   git push
   ```
2. [vercel.com/new](https://vercel.com/new) → import
   `Anucodex21/datoryx-operator`.
3. Vercel reads `vercel.json` automatically (it's a static site, no build
   step). Deploy.
4. You get a URL like `https://datoryx-operator.vercel.app` — this is
   your live, public dashboard.

Optional tidy-up: in Render's env vars, set
`DATORYX_CORS_ORIGINS=https://datoryx-operator.vercel.app` so the backend
only accepts requests from your actual frontend instead of any origin.

---

## Quick reference — where everything is

| I want to... | Command / file |
|---|---|
| Add an API key | edit `.env`, or Render's dashboard env vars for the live version |
| Run locally | `docker compose up --build`, open localhost:8000 |
| Run as MCP server | `cd backend && python -m datoryx_mcp.mcp_server` |
| Train once | `bash backend/scripts/auto_train.sh 20000` |
| Train automatically forever | `bash backend/scripts/setup_cron.sh` |
| Pause/resume auto-training | `pause_training.sh` / `resume_training.sh` |
| Push code changes live | `git add -A && git commit -m "..." && git push` — Render/Vercel both auto-redeploy on push |
| Check backend is alive | `curl https://<your-render-url>/health` |
