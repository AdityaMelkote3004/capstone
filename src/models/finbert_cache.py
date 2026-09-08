"""FinBERT embedding with a disk cache, shared by Experiment 7's macro and
micro channels.

The 40-run ablation ladder must not re-embed text 40 times, and FinBERT on CPU
is the experiment's single largest fixed cost, so every embedding is computed
once and cached by a stable key.

`cap_per_day` is not an optimisation but a feasibility requirement: the macro
channel projects ~7,600 headlines per trading day, which at BERT-base CPU
throughput would take tens of hours to embed.
"""

import collections
import os

import torch

MODEL_NAME = "ProsusAI/finbert"
_model = None
_tokenizer = None


def _load_model():
    """Lazily loads FinBERT so importing this module stays cheap in tests."""
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModel, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model, _tokenizer


def cap_per_day(rows: list, n: int, cutoff_key: str = "timestamp") -> list:
    """Keeps at most `n` rows per calendar day, the most recent ones by
    timestamp. Selection uses only timestamps, never labels."""
    by_day = collections.defaultdict(list)
    for row in rows:
        by_day[row[cutoff_key].strftime("%Y-%m-%d")].append(row)

    kept = []
    for day_rows in by_day.values():
        day_rows.sort(key=lambda r: r[cutoff_key])
        kept.extend(day_rows[-n:])
    return kept


def embed_texts(texts: list, batch_size: int = 32) -> torch.Tensor:
    """Mean-pools FinBERT's last hidden state over non-padding tokens,
    returning (len(texts), 768). Mean-pooling rather than [CLS] because
    FinBERT's [CLS] is tuned for its sentiment head, not general similarity --
    and Task 4 builds a cosine-similarity graph on these vectors."""
    if not texts:
        return torch.zeros((0, 768))

    model, tokenizer = _load_model()
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[start:start + batch_size]]
            encoded = tokenizer(batch, padding=True, truncation=True,
                                max_length=128, return_tensors="pt")
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            out.append(pooled)
    return torch.cat(out, dim=0)


class EmbeddingCache:
    """Keyed store of 768-d vectors, persisted with torch.save."""

    def __init__(self, store: dict = None):
        self._store = store or {}

    def get(self, key: str):
        return self._store.get(key)

    def put(self, key: str, tensor: torch.Tensor) -> None:
        self._store[key] = tensor

    def __len__(self) -> int:
        return len(self._store)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._store, path)

    @classmethod
    def load(cls, path: str) -> "EmbeddingCache":
        """Missing file yields an empty cache, so a first run and a resumed run
        take the same code path."""
        if not os.path.exists(path):
            return cls()
        return cls(torch.load(path, weights_only=True))
