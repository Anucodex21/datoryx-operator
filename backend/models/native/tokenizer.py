"""DATORYX Native Tokenizer - a minimal character-level tokenizer.

No external dependency (no tiktoken/sentencepiece) so the native model has
zero extra install cost beyond torch itself. Character-level keeps the
vocab small (~100 symbols for source code + English text), which matters
a lot when the model itself is tiny and training runs on CPU.

The vocab is built once from the training corpus and saved alongside the
model checkpoint (vocab.json), so tokenization at inference time is exactly
reproducible regardless of what text the model happens to see later.
"""
import json
import os
from typing import Dict, List


class CharTokenizer:
    def __init__(self, stoi: Dict[str, int], itos: Dict[int, str]):
        self.stoi = stoi
        self.itos = itos

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @classmethod
    def build_from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return cls(stoi, itos)

    def encode(self, text: str) -> List[int]:
        # Unknown characters (not seen during vocab-building) are dropped
        # rather than crashing - keeps inference robust against whatever
        # a caller happens to type.
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids if i in self.itos)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        stoi = data["stoi"]
        itos = {int(v): k for k, v in stoi.items()}
        return cls(stoi, itos)


def build_corpus(root_dir: str, extensions=(".py", ".md")) -> str:
    """Concatenate every source/doc file under root_dir into one training
    corpus. Deliberately trains on DATORYX's own codebase and docs rather
    than requiring any external dataset download - the point of the
    "native" provider is to be genuinely self-contained: no cloud API,
    no external model download, not even an external training corpus."""
    chunks = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in (
            ".venv", "__pycache__", "node_modules", ".git", "models"
        )]
        for fname in filenames:
            if fname.endswith(extensions):
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        chunks.append(f.read())
                except OSError:
                    continue
    return "\n\n".join(chunks)


def build_extra_corpus(extra_dir: str) -> str:
    """Read arbitrary text you supply yourself - any file under extra_dir,
    any extension (.txt, .py, .md, .csv, notes, chat logs, whatever) - and
    concatenate it into extra training text, separate from build_corpus()'s
    repo-only scan. This is the hook for "train it on my own text" rather
    than only DATORYX's own source code: point --extra-text-dir at a
    folder of anything you want the model to learn from, and it gets
    appended to the corpus alongside the repo's own code/docs.

    No extension filtering here on purpose - if you put a file in the
    folder, you want it included. Binary files are skipped safely (opened
    as UTF-8 with errors="ignore", so a stray non-text file just
    contributes garbage-but-harmless bytes rather than crashing the run).
    """
    if not extra_dir or not os.path.isdir(extra_dir):
        return ""
    chunks = []
    for dirpath, dirnames, filenames in os.walk(extra_dir):
        dirnames[:] = [d for d in dirnames if d not in (
            ".venv", "__pycache__", "node_modules", ".git"
        )]
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if content.strip():
                    chunks.append(content)
            except OSError:
                continue
    return "\n\n".join(chunks)
