# Deploy Guide

Everything is prepped (git repo initialized, Dockerfile, render.yaml,
vercel.json) — these are the exact commands to actually put it live under
your own GitHub, Render, and Vercel accounts.

## 1. Push to GitHub (Anucodex21)

```bash
cd datoryx-operator
# repo is already git-initialized with one commit - just add the remote
gh repo create Anucodex21/datoryx-operator --public --source=. --remote=origin
git push -u origin main
```

No `gh` CLI? Do it from the website instead:
1. Go to github.com/new, create a repo named `datoryx-operator` under Anucodex21, don't initialize with a README.
2. Then:
```bash
cd datoryx-operator
git remote add origin https://github.com/Anucodex21/datoryx-operator.git
git push -u origin main
```

## 2. Deploy the backend on Render

1. Go to [render.com](https://render.com) → **New** → **Blueprint**
2. Connect your `Anucodex21/datoryx-operator` GitHub repo — Render reads `render.yaml` automatically and sets up the Docker service.
3. It'll ask for the env vars marked `sync: false` in `render.yaml` — paste in whichever LLM provider key you have (`GROQ_API_KEY` is the easiest free one to get). You can leave them blank and the app still runs (simulated-response mode).
4. Deploy. You'll get a URL like `https://datoryx-operator-api.onrender.com`.
5. Confirm it's up: `curl https://datoryx-operator-api.onrender.com/health` should return `{"status":"ok",...}`.

Free-tier Render services sleep after inactivity and take ~30s to wake on the first request — normal, not a bug.

## 3. Deploy the frontend on Vercel

1. Before deploying, point the dashboard at your live backend: edit `frontend/index.html`, find `window.DATORYX_API_BASE = "";` near the top of `<body>`, and set it to your Render URL:
   ```html
   window.DATORYX_API_BASE = "https://datoryx-operator-api.onrender.com";
   ```
   Commit and push that change.
2. Go to [vercel.com/new](https://vercel.com/new) → import `Anucodex21/datoryx-operator`.
3. Vercel reads `vercel.json` (output directory = `frontend`) automatically — no build command needed, it's a static file. Deploy.
4. You'll get a URL like `https://datoryx-operator.vercel.app` — that's your live dashboard, talking to the Render backend.

Also update the backend's CORS to only allow your Vercel domain (optional but tidier than the `*` default):
in Render's env vars, set `DATORYX_CORS_ORIGINS=https://datoryx-operator.vercel.app`.

## 4. Run it locally on your laptop

```bash
git clone https://github.com/Anucodex21/datoryx-operator.git
cd datoryx-operator
cp .env.example .env
# edit .env, add whichever LLM provider key you have (optional)

docker compose up --build
```

Open **http://localhost:8000** — full dashboard, same code that's running on Render/Vercel, entirely on your machine. No Docker? see the README's non-Docker `uvicorn` instructions.

To also run the MCP server locally (for connecting Claude Desktop / Claude Code to your 21 agents):
```bash
cd backend
python -m datoryx_mcp.mcp_server
```

## 5. Automate DAXRO 1.0 training (fully hands-off, runs every night)

One-time setup:

```bash
cd backend
pip install torch
bash scripts/setup_cron.sh
```

That installs a cron job (default: 2:30 AM daily) that runs
`scripts/nightly_train.sh`, which:
- reads how many steps the checkpoint is currently trained to
- adds 4,000 more steps as tonight's target (configurable: `bash scripts/nightly_train.sh 8000`)
- calls `auto_train.sh` toward that target, which itself resumes from the exact checkpoint on disk
- stops growing once it hits `NIGHTLY_MAX_STEPS` (default 100,000) so it doesn't run forever unattended

Custom schedule:
```bash
bash scripts/setup_cron.sh "0 4 * * *"   # 4:00 AM instead of 2:30 AM
```

Check it's installed: `crontab -l`
Watch it run: `tail -f backend/models/native/checkpoint/nightly_train.log`
Remove it: `crontab -e`, delete the `# datoryx-nightly-train` line and the one after it

**This needs the machine to be on and awake at the scheduled time.** If
it's a laptop that sleeps overnight, either pick a schedule for when it's
usually on, or move this to a server (a Render cron job pointed at the
same `nightly_train.sh`, or any always-on VPS) instead.

## Order to do this in

GitHub push → Render (backend) → grab the Render URL → put it in
`frontend/index.html` → push again → Vercel (frontend). Vercel needs the
backend URL to exist first, so backend goes live before frontend.
