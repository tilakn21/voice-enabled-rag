"""Sentence encoder.

`intfloat/multilingual-e5-small` — 384 dims, 12 layers, and trained on 100+
languages, which matters because MSMARCO-XI is 13 Indic languages plus English
and an English-only encoder scores near chance on Tamil or Odia queries.

Three things keep query encoding inside the latency budget:
  * int8 dynamic quantisation of the Linear layers (~1.8x faster on CPU)
  * a capped sequence length (queries are short; padding to 512 is wasted work)
  * a bounded thread pool, so we don't thrash on a small container

`encode_spans` exists for the late-chunking strategy: it runs one forward pass
over a whole document and mean-pools token embeddings per character span, so
each chunk vector carries document-level context.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


class Encoder:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        *,
        max_tokens: int = 192,
        quantize: bool = True,
        threads: int = 4,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        cache_size: int = 512,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        torch.set_num_threads(max(1, threads))
        torch.set_grad_enabled(False)

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        if quantize:
            try:
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("encoder: int8 dynamic quantisation enabled")
            except Exception as exc:  # noqa: BLE001
                logger.warning("encoder: quantisation unavailable (%s), using fp32", exc)

        self.model = model
        self.dim = int(getattr(model.config, "hidden_size", 384))

        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self.cache_enabled = True

    # ------------------------------------------------------------------
    def warmup(self, n: int = 3) -> None:
        """Force lazy kernel init so the first real request isn't the slow one."""
        for _ in range(n):
            self.encode_queries(["warmup query"], use_cache=False)

    # ------------------------------------------------------------------
    def _forward(self, texts: Sequence[str], max_length: int | None = None) -> np.ndarray:
        torch = self._torch
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length or self.max_tokens,
            return_tensors="pt",
        )
        with torch.inference_mode():
            out = self.model(**batch)
        hidden = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    def encode_passages(
        self, texts: Sequence[str], batch_size: int = 128, max_length: int | None = None
    ) -> np.ndarray:
        """`max_length` lets callers encode known-short text (single sentences)
        without padding the batch out to the full passage window."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefixed = [f"{self.passage_prefix}{t}" for t in texts]
        chunks = []
        for i in range(0, len(prefixed), batch_size):
            chunks.append(self._forward(prefixed[i : i + batch_size], max_length=max_length))
        return np.vstack(chunks)

    def encode_queries(self, texts: Sequence[str], use_cache: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        use_cache = use_cache and self.cache_enabled
        if use_cache:
            hits: dict[int, np.ndarray] = {}
            misses: list[int] = []
            with self._lock:
                for i, t in enumerate(texts):
                    vec = self._cache.get(t)
                    if vec is None:
                        misses.append(i)
                    else:
                        self._cache.move_to_end(t)
                        hits[i] = vec
            if not misses:
                return np.stack([hits[i] for i in range(len(texts))])
            computed = self._forward([f"{self.query_prefix}{texts[i]}" for i in misses])
            with self._lock:
                for slot, i in enumerate(misses):
                    self._cache[texts[i]] = computed[slot]
                    self._cache.move_to_end(texts[i])
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            for slot, i in enumerate(misses):
                hits[i] = computed[slot]
            return np.stack([hits[i] for i in range(len(texts))])

        return self._forward([f"{self.query_prefix}{t}" for t in texts])

    def encode_query(self, text: str, use_cache: bool = True) -> np.ndarray:
        return self.encode_queries([text], use_cache=use_cache)[0]

    # ------------------------------------------------------------------
    def encode_spans(self, document: str, spans: Sequence[tuple[int, int]]) -> np.ndarray:
        """Late chunking: one forward pass over the document, mean-pool per span.

        Falls back to independent per-span encoding if the tokenizer can't give
        offset mappings (slow tokenizers) or the doc overflows the window.
        """
        torch = self._torch
        text = f"{self.passage_prefix}{document}"
        offset_shift = len(self.passage_prefix)

        try:
            batch = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
        except (TypeError, NotImplementedError):
            return self.encode_passages([document[s:e] for s, e in spans])

        offsets = batch.pop("offset_mapping")[0].numpy()
        with torch.inference_mode():
            out = self.model(**batch)
        hidden = out.last_hidden_state[0]  # (seq, dim)
        attn = batch["attention_mask"][0].numpy().astype(bool)

        # Map token offsets back into original-document coordinates.
        starts = offsets[:, 0] - offset_shift
        ends = offsets[:, 1] - offset_shift
        # Special tokens have zero-width offsets; drop them.
        real = attn & (ends > starts)

        vectors = []
        for s, e in spans:
            sel = real & (starts < e) & (ends > s)
            if not sel.any():
                vectors.append(self.encode_passages([document[s:e]])[0])
                continue
            idx = torch.from_numpy(np.where(sel)[0])
            pooled = hidden.index_select(0, idx).mean(dim=0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=0)
            vectors.append(pooled.cpu().numpy().astype(np.float32))
        return np.stack(vectors)
