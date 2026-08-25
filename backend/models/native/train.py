"""Train DATORYX's native from-scratch model.

Zero external downloads: the training corpus is DATORYX's own source code
and docs (see tokenizer.build_corpus), and the architecture is
hand-written in models/native/model.py - no pretrained weights loaded
from anywhere. This is what makes the resulting checkpoint genuinely
"trained by you" rather than a downloaded model running locally.

Usage:
    python -m models.native.train
    python -m models.native.train --steps 3000 --n-layer 6 --n-embd 192

Writes models/native/checkpoint/model.pt and vocab.json. LLMManager
auto-detects a checkpoint there the same way it auto-detects a GGUF file
for the llama.cpp-backed local provider (see models/llm/providers.py:
NativeTransformerProvider).
"""
import argparse
import json
import os
import time

import torch

from .model import DatoryxNativeGPT, NativeConfig
from .tokenizer import CharTokenizer, build_corpus, build_extra_corpus


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=os.path.join(os.path.dirname(__file__), "..", ".."))
    parser.add_argument("--extra-text-dir", default=None,
                         help="Optional folder of your own text (any file type/extension) to "
                              "train on in addition to the repo's own code/docs. Everything "
                              "under this folder is read as plain text and appended to the "
                              "training corpus - notes, chat logs, articles, whatever you want "
                              "the model to learn from.")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "checkpoint"))
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--resume", action="store_true",
                         help="Resume from an existing checkpoint in --out-dir instead of starting fresh.")
    parser.add_argument("--patience", type=int, default=0,
                         help="Early stopping: stop once val_loss hasn't improved on the best-seen "
                              "value for this many consecutive evals. 0 (default) disables early "
                              "stopping and trains the full --steps regardless - matches the "
                              "original behavior. Try 15-25 to auto-stop near the generalization "
                              "point instead of guessing a step count (this repo's shipped "
                              "checkpoint was trained without this and overfit badly past step "
                              "~5,400 as a result - see README).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)

    print(f"[native-train] building corpus from {args.repo_root} ...")
    text = build_corpus(args.repo_root)
    print(f"[native-train] repo corpus size: {len(text):,} chars")

    if args.extra_text_dir:
        extra_text = build_extra_corpus(args.extra_text_dir)
        if extra_text:
            print(f"[native-train] extra corpus from {args.extra_text_dir}: {len(extra_text):,} chars")
            text = text + "\n\n" + extra_text
        else:
            print(f"[native-train] WARNING: --extra-text-dir {args.extra_text_dir} "
                  "had no readable text - continuing with just the repo corpus.")
    print(f"[native-train] total corpus size: {len(text):,} chars")

    ckpt_path = os.path.join(args.out_dir, "model.pt")
    best_ckpt_path = os.path.join(args.out_dir, "model_best.pt")
    vocab_path = os.path.join(args.out_dir, "vocab.json")
    log_path = os.path.join(args.out_dir, "train_log.json")

    resuming = args.resume and os.path.isfile(ckpt_path) and os.path.isfile(vocab_path)
    if resuming:
        print(f"[native-train] resuming from {ckpt_path}")
        tok = CharTokenizer.load(vocab_path)
    else:
        tok = CharTokenizer.build_from_text(text)
    print(f"[native-train] vocab size: {tok.vocab_size}")

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    if resuming:
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = NativeConfig(**ckpt["config"])
        model = DatoryxNativeGPT(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        start_step = ckpt.get("step", 0)
        log = ckpt.get("log", [])
        if os.path.isfile(log_path):
            try:
                with open(log_path) as f:
                    log = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
    else:
        cfg = NativeConfig(
            vocab_size=tok.vocab_size, block_size=args.block_size,
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        )
        model = DatoryxNativeGPT(cfg).to(device)
        start_step = 0
        log = []
    print(f"[native-train] model params: {model.num_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if resuming and "optimizer_state" in ckpt:
        opt.load_state_dict(ckpt["optimizer_state"])

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # Best-seen val_loss so far (across this run and any resumed history),
    # used for both model_best.pt tracking and --patience early stopping.
    best_val_loss = min((e["val_loss"] for e in log), default=float("inf"))
    evals_since_improvement = 0

    target_step = start_step + args.steps
    for step in range(start_step + 1, target_step + 1):
        xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_interval == 0 or step == start_step + 1 or step == target_step:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, args.block_size, args.batch_size, device)
                _, val_loss = model(vx, vy)
            model.train()
            elapsed = time.time() - t0
            entry = {
                "step": step, "train_loss": round(loss.item(), 4),
                "val_loss": round(val_loss.item(), 4), "elapsed_sec": round(elapsed, 1),
            }
            log.append(entry)
            print(f"[native-train] step {step}/{target_step}  "
                  f"train_loss={entry['train_loss']}  val_loss={entry['val_loss']}  "
                  f"elapsed={entry['elapsed_sec']}s")

            # Checkpoint after every eval, not just at the end - a training
            # run on limited wall-clock time (e.g. chained CPU sessions)
            # needs to survive being interrupted mid-run without losing
            # progress, the same way a real training job would checkpoint
            # against a preemptible/spot instance.
            checkpoint_payload = {
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "config": cfg.__dict__,
                "step": step,
                "log": log,
            }
            torch.save(checkpoint_payload, ckpt_path)
            tok.save(vocab_path)
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)

            if entry["val_loss"] < best_val_loss:
                best_val_loss = entry["val_loss"]
                evals_since_improvement = 0
                torch.save(checkpoint_payload, best_ckpt_path)
                print(f"[native-train]   ^ new best val_loss ({best_val_loss}) - saved to {best_ckpt_path}")
            else:
                evals_since_improvement += 1
                if args.patience and evals_since_improvement >= args.patience:
                    print(f"[native-train] early stopping: val_loss hasn't improved on "
                          f"{best_val_loss} for {evals_since_improvement} evals "
                          f"(--patience {args.patience}). Best checkpoint is in {best_ckpt_path}.")
                    break

    print(f"[native-train] done. checkpoint written to {ckpt_path} (last step {step})")
    print(f"[native-train] best val_loss={best_val_loss} - that checkpoint is saved separately at {best_ckpt_path}")

    # Quick qualitative sample so the log itself shows real generated output.
    model.eval()
    prompt = "def "
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=200, temperature=0.8, top_k=40)
    sample = tok.decode(out[0].tolist())
    print("[native-train] sample generation:")
    print(sample)
    with open(os.path.join(args.out_dir, "sample_generation.txt"), "w") as f:
        f.write(sample)


if __name__ == "__main__":
    main()
