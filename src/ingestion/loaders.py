"""Loaders for heterogeneous enterprise sources: PDF, Slack JSON export, Markdown.

Every loader emits a list of RawDoc, a common envelope carrying provenance
(source system, path, timestamp, region hints). Provenance is not decoration:
downstream conflict resolution depends on knowing WHERE and WHEN a statement
came from.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


@dataclass
class RawDoc:
    doc_id: str
    text: str
    source_system: str          # "pdf" | "slack" | "manual"
    source_path: str
    title: str = ""
    timestamp: str | None = None   # ISO8601 where known
    region: str | None = None      # "UK" | "India" | "Global" | None
    extra: dict = field(default_factory=dict)


_REGION_HINTS = {
    "uk": "UK", "united kingdom": "UK", "britain": "UK",
    "india": "India", "global": "Global",
}


def _infer_region(text: str) -> str | None:
    lowered = text.lower()
    for hint, region in _REGION_HINTS.items():
        if hint in lowered:
            return region
    return None


def load_pdfs(pdf_dir: str | Path) -> list[RawDoc]:
    docs: list[RawDoc] = []
    for path in sorted(Path(pdf_dir).glob("*.pdf")):
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
        text = "\n".join(pages)
        text = re.sub(r"[ \t]+", " ", text)          # collapse spacing artifacts
        text = re.sub(r"\n{3,}", "\n\n", text)
        title = text.strip().split("\n")[0][:120] if text.strip() else path.stem
        docs.append(RawDoc(
            doc_id=f"pdf::{path.stem}",
            text=text,
            source_system="pdf",
            source_path=str(path),
            title=title,
            region=_infer_region(title) or _infer_region(text[:500]),
        ))
    return docs


_NOISE_CHANNELS = {"#random", "#general", "#watercooler", "#social"}
_POLICY_KEYWORDS = ("policy", "leave", "vpn", "probation", "expense", "notice",
                    "remote", "hybrid", "security", "mandatory", "effective")


def load_slack_export(json_path: str | Path, min_signal_len: int = 40) -> list[RawDoc]:
    """One RawDoc per message. Channel-aware noise filtering:
    - announcement/policy channels: keep everything
    - known noise channels (#random etc.): keep ONLY if a policy keyword appears
    - other channels: drop short chatter below min_signal_len"""
    messages = json.loads(Path(json_path).read_text())
    docs: list[RawDoc] = []
    for i, msg in enumerate(messages):
        text = msg.get("text", "").strip()
        channel = msg.get("channel", "")
        if not text:
            continue
        is_announcement = "announce" in channel or "policy" in channel
        if not is_announcement:
            lowered = text.lower()
            if channel in _NOISE_CHANNELS and not any(k in lowered for k in _POLICY_KEYWORDS):
                continue
            if len(text) < min_signal_len:
                continue
        docs.append(RawDoc(
            doc_id=f"slack::{channel}::{i}",
            text=f"[Slack {channel} | {msg.get('user','?')} | {msg.get('ts','?')}] {text}",
            source_system="slack",
            source_path=str(json_path),
            title=f"Slack message in {channel}",
            timestamp=msg.get("ts"),
            region=_infer_region(text),
            extra={"channel": channel, "user": msg.get("user")},
        ))
    return docs


def load_manuals(manual_dir: str | Path) -> list[RawDoc]:
    docs: list[RawDoc] = []
    for path in sorted(Path(manual_dir).glob("*.md")):
        text = path.read_text()
        title_match = re.match(r"#\s*(.+)", text)
        title = title_match.group(1).strip() if title_match else path.stem
        docs.append(RawDoc(
            doc_id=f"manual::{path.stem}",
            text=text,
            source_system="manual",
            source_path=str(path),
            title=title,
            region=_infer_region(title) or _infer_region(text[:300]),
        ))
    return docs


def load_all(data_root: str | Path = "data/raw") -> list[RawDoc]:
    root = Path(data_root)
    docs: list[RawDoc] = []
    docs += load_pdfs(root / "pdfs")
    slack_files = list((root / "slack").glob("*.json"))
    for sf in slack_files:
        docs += load_slack_export(sf)
    docs += load_manuals(root / "manuals")
    return docs
