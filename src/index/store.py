"""Hybrid retrieval index.

Two retrievers behind one interface:
  - BM25 (lexical): zero-dependency-heavy, deterministic, great for policy text
    where exact terms ("probation", "carry-over") matter.
  - Dense (semantic): optional sentence-transformers embeddings for paraphrase
    robustness ("how many days can I WFH" -> remote work policy).

Scores are fused with Reciprocal Rank Fusion (RRF), which needs no score
calibration between the two systems - a practical trick worth knowing for
production RAG.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.ingestion.chunking import Chunk

_TOKEN_SPLIT = lambda s: [t.lower() for t in s.split()]


class HybridIndex:
    def __init__(self, chunks: list[Chunk], use_dense: bool = False,
                 dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.bm25 = BM25Okapi([_TOKEN_SPLIT(c.text) for c in chunks])
        self.dense = None
        self.embeddings = None
        if use_dense:
            from sentence_transformers import SentenceTransformer  # lazy import
            self.dense = SentenceTransformer(dense_model)
            self.embeddings = self.dense.encode(
                [c.text for c in chunks], normalize_embeddings=True)

    # ---------- retrieval ----------
    def _bm25_ranks(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(_TOKEN_SPLIT(query))
        return sorted(range(len(scores)), key=lambda i: -scores[i])

    def _dense_ranks(self, query: str) -> list[int]:
        q = self.dense.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ q
        return sorted(range(len(sims)), key=lambda i: -sims[i])

    def search(self, query: str, k: int = 5, rrf_k: int = 60) -> list[Chunk]:
        """Reciprocal Rank Fusion over available retrievers."""
        rankings = [self._bm25_ranks(query)]
        if self.dense is not None:
            rankings.append(self._dense_ranks(query))
        fused: dict[int, float] = {}
        for ranks in rankings:
            for rank_pos, idx in enumerate(ranks):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank_pos + 1)
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [self.chunks[i] for i in top]

    # ---------- persistence ----------
    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks}, f)
        # human-inspectable sidecar
        with open(path.with_suffix(".jsonl"), "w") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c)) + "\n")

    @classmethod
    def load(cls, path: str | Path, use_dense: bool = False) -> "HybridIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["chunks"], use_dense=use_dense)
