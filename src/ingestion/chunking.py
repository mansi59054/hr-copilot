"""Structure-aware chunking.

Policy documents have natural units (sections). Chunking on section boundaries
keeps each chunk self-contained, which matters twice downstream: retrieval
precision, and citation quality (a chunk that starts mid-sentence makes a bad
citation).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .loaders import RawDoc

_SECTION_RE = re.compile(
    r"(?=Section\s+\d+(?:\.\d+)*)|(?=^#{1,3}\s)", re.MULTILINE
)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    source_system: str
    title: str
    region: str | None
    timestamp: str | None
    meta: dict = field(default_factory=dict)


def chunk_doc(doc: RawDoc, max_chars: int = 900, overlap: int = 120) -> list[Chunk]:
    # Slack messages are already atomic
    if doc.source_system == "slack":
        return [Chunk(
            chunk_id=f"{doc.doc_id}::c0", text=doc.text, doc_id=doc.doc_id,
            source_system=doc.source_system, title=doc.title,
            region=doc.region, timestamp=doc.timestamp, meta=doc.extra,
        )]

    pieces = [p.strip() for p in _SECTION_RE.split(doc.text) if p and p.strip()]
    if not pieces:
        pieces = [doc.text.strip()]

    # Merge tiny pieces forward, split oversized ones with overlap
    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) + len(piece) < max_chars // 2:
            merged[-1] = merged[-1] + "\n" + piece
        else:
            merged.append(piece)

    chunks: list[Chunk] = []
    idx = 0
    for piece in merged:
        start = 0
        while start < len(piece):
            window = piece[start:start + max_chars]
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}::c{idx}", text=window, doc_id=doc.doc_id,
                source_system=doc.source_system, title=doc.title,
                region=doc.region, timestamp=doc.timestamp,
            ))
            idx += 1
            if start + max_chars >= len(piece):
                break
            start += max_chars - overlap
    return chunks


def chunk_all(docs: list[RawDoc]) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_doc(d))
    return out
