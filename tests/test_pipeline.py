import sys
sys.path.insert(0, ".")

from src.ingestion.loaders import load_all
from src.ingestion.chunking import chunk_all
from src.index.store import HybridIndex


def test_ingestion_covers_all_source_systems():
    docs = load_all("data/raw")
    systems = {d.source_system for d in docs}
    assert systems == {"pdf", "slack", "manual"}


def test_slack_noise_filtered():
    docs = load_all("data/raw")
    texts = " ".join(d.text for d in docs)
    assert "lol" not in texts, "short #random chatter should be filtered"


def test_chunks_carry_provenance():
    chunks = chunk_all(load_all("data/raw"))
    assert all(c.doc_id and c.source_system for c in chunks)
    assert any(c.region == "UK" for c in chunks)
    assert any(c.region == "India" for c in chunks)


def test_retrieval_surfaces_conflicting_sources():
    """The probation query must retrieve BOTH the Slack update and the PDF,
    otherwise the synthesizer can never resolve the conflict."""
    index = HybridIndex(chunk_all(load_all("data/raw")))
    ids = [c.chunk_id for c in index.search("probation period india engineering", k=6)]
    assert any(i.startswith("slack::") for i in ids)
    assert any(i.startswith("pdf::india_hr_policy_2025") for i in ids)


def test_region_metadata_on_slack():
    docs = load_all("data/raw")
    slack_docs = [d for d in docs if d.source_system == "slack"]
    assert any(d.region == "UK" for d in slack_docs)
