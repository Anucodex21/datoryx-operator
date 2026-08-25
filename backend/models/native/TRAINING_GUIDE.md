# Training the native model further — where and how

The checkpoint that ships in `checkpoint/` was trained for 1,260 steps on
a single CPU core. The pipeline is real and scales — more steps, a
bigger corpus, and a bigger model all genuinely improve output quality.
This is where to actually do that.

## Option A: keep going on your own laptop (free, slow)

No setup beyond what's already here.

```bash
cd backend
pip install torch
python -m models.native.train --steps 5000 --resume
```

`--resume` picks up from the shipped checkpoint instead of starting
over. Run this command again any time — it always continues from
wherever it left off. Expect roughly 0.5–1s/step on a modern laptop CPU,
so 5,000 steps is on the order of an hour.

## Option B: Google Colab (free GPU, best option for you)

Colab gives you a free T4 GPU (12-hour session limit on the free tier).
A GPU makes this dramatically faster — the same 1,260 steps that took
several minutes on CPU would take well under a minute on a T4.

1. Go to https://colab.research.google.com, new notebook.
2. Runtime → Change runtime type → **T4 GPU**.
3. Upload just the `models/native/` folder (and nothing else needed —
   the corpus builder needs `.py`/`.md` files to train on, so also zip
   and upload `backend/` if you want it training on the real DATORYX
   corpus rather than a smaller sample).
4. In a cell:
   ```python
   !pip install torch
   !python -m models.native.train --steps 20000 --repo-root /content/backend --out-dir /content/backend/models/native/checkpoint --resume
   ```
5. Download the resulting `checkpoint/` folder (`model.pt`,
   `vocab.json`, `train_log.json`) and drop it back into
   `backend/models/native/checkpoint/` in your local project, replacing
   the old one.

## Option C: Kaggle Notebooks (free GPU, no time limit hassle)

Similar to Colab but with a more generous free GPU quota (30 hrs/week on
a P100/T4) and no 12-hour hard cutoff per session.

1. https://kaggle.com/code → New Notebook.
2. Settings → Accelerator → GPU T4 x2 (or P100).
3. Upload `backend/` as a Kaggle Dataset, or `!git clone` if you push
   this repo to GitHub first.
4. Same training command as Colab, adjusted for Kaggle's paths
   (`/kaggle/working/...`).
5. Download the checkpoint from the notebook's output panel.

## Option D: RunPod / Vast.ai (cheap dedicated GPU, best if scaling up seriously)

If you want to go well beyond what a free-tier GPU session supports —
much bigger model, much bigger corpus, many hours of training — rent a
GPU pod. This is the same infra your NEXUS-GPT `runpod_train.sh` script
already targets, so the workflow will feel familiar:

1. https://runpod.io (or vast.ai) → deploy a pod, e.g. an RTX 3090/4090
   (cheapest GPUs that still train fast — no need for an A100 at this
   model size).
2. SSH in, clone/upload the repo.
3. `pip install torch` (comes preinstalled on most RunPod PyTorch
   templates) `&& python -m models.native.train --steps 100000 --n-layer 8 --n-embd 256 --resume`.
4. Download the checkpoint when done, or push it to a bucket/GitHub
   release and pull it down locally.
5. **Stop the pod when you're done** — you're billed by the hour.

## What actually moves the needle on output quality

In rough order of impact, cheapest first:

1. **More steps.** 1,260 → 20,000+ steps alone will take it from "learned
   Python syntax shapes" to noticeably more coherent structure.
2. **Bigger corpus.** `--repo-root` currently points at just this repo
   (680K chars). Point it at a bigger `.py`/`.md` collection — e.g. a
   folder with a few of your other projects (NEXUS-GPT, Typing HUB,
   Jarvis, DataForge) — for more variety to learn from, still with zero
   external downloads.
3. **Bigger model.** `--n-layer 6 --n-embd 256` (up from the shipped
   `4`/`128`) roughly doubles parameter count — meaningfully more
   capacity, at the cost of slower steps. Only worth it once you're
   training on GPU; on CPU it'll be painfully slow.
4. **A real (sub-word) tokenizer** instead of character-level would help
   a lot at scale — this repo's `tokenizer.py` is intentionally
   dependency-free for simplicity, but swapping in a BPE tokenizer
   (e.g. via the `tokenizers` library) is the next real upgrade once
   you're training seriously.

Every one of these is a config flag or a small code change away — the
pipeline itself doesn't need to change to scale up.
