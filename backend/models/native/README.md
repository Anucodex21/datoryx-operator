# DATORYX Native Model

A small decoder-only transformer, **built and trained entirely from
scratch inside this repo** - no pretrained weights, no external dataset
download, no cloud API. This is what makes it different from the
`local` provider (which runs a *downloaded* GGUF model via
llama-cpp-python): the native model's weights were produced by
`models/native/train.py` running on this machine, on this repo's own
source code and docs.

## What's actually in here

- `tokenizer.py` - a minimal character-level tokenizer (no dependency on
  tiktoken/sentencepiece) and `build_corpus()`, which concatenates every
  `.py`/`.md` file in the repo into the training corpus. Zero external
  data source - the model is trained on DATORYX's own codebase.
- `model.py` - `DatoryxNativeGPT`: multi-head causal self-attention with
  rotary positional embeddings (RoPE), pre-LN transformer blocks, and
  weight tying between the token embedding and the output head. Same
  core architectural choices as the earlier from-scratch NEXUS-GPT
  project, at a much smaller scale.
- `train.py` - the training loop. Supports `--resume` so a run can be
  checkpointed and continued across separate sessions (useful on CPU,
  where a full run takes longer than any single sitting).
- `checkpoint/` - the actual trained weights (`model.pt`), the tokenizer
  vocab (`vocab.json`), and the full training log (`train_log.json`).

`models/llm/providers.py::NativeTransformerProvider` wires this model
into DATORYX's normal provider interface, so it's selectable exactly
like `local`, `groq`, `openai`, etc. `LLMManager` auto-registers it if a
checkpoint exists in `checkpoint/` (see `_try_register_native` in
`models/llm/manager.py`) - no config needed.

## What this run actually proves

The checkpoint shipped in `checkpoint/` was trained for **1,260 steps**
on a **680,173-character** corpus (this repo's own `.py`/`.md` files),
single CPU core, in chained ~80-second sessions. The loss curve is real:

| step | train_loss | val_loss |
|------|-----------|----------|
| 1    | 4.60      | 4.13     |
| 1260 | 1.23      | 1.32     |

That's genuine evidence the pipeline works end-to-end: real forward
passes, real backprop, real gradient descent, loss actually decreasing.
At this point the model has learned real structure - Python keywords,
`def `/`class `/`self.` patterns, indentation, dict/JSON-like shapes -
but **not** coherent prose or working code. A representative sample at
this checkpoint (temperature 0.7, top-k 30):

```
prompt: "def "
output: "def test_this_headers, and boals wheth althery the the
         persister.\n        self._workflow(self.reasoning) a value
         the per-loid fake the lown the * in with "
```

## What this is *not*

- **Not a chatbot.** It has no chat template and no instruction-tuning -
  it's a raw language model continuing whatever text it's given, the
  same way GPT-2 (pre-ChatGPT) worked.
- **Not comparable to a downloaded pretrained model or a cloud API.**
  Llama/Mistral/GPT-class models are trained on trillions of tokens
  across a GPU cluster for weeks. This is ~800K parameters trained on
  680K characters for a few CPU-minutes. The gap is enormous and
  expected - the point of this model isn't output quality, it's proving
  a real, working, from-scratch training pipeline exists and runs.
- **Not production-ready.** If you need DATORYX to actually hold a
  coherent conversation, use the `local` provider with a real downloaded
  GGUF (`python scripts/setup_local_model.py`) or a cloud provider.
  `native` is the "fully self-built, zero downloads" option, not the
  "best quality" option.

## Training more

```bash
pip install torch
python -m models.native.train --steps 5000                  # fresh run
python -m models.native.train --steps 5000 --resume          # continue
```

More steps, a bigger corpus (point `--repo-root` at more text), and a
larger `--n-layer`/`--n-embd` will all improve output quality - this is
a genuine, scalable pipeline, just run here at a scale that fits a
single CPU session.
