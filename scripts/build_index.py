"""CLI: rebuild the index from data/raw. Run after any data change."""
import sys
sys.path.insert(0, ".")

from src.ingestion.loaders import load_all
from src.ingestion.chunking import chunk_all
from src.index.store import HybridIndex

if __name__ == "__main__":
    docs = load_all("data/raw")
    chunks = chunk_all(docs)
    index = HybridIndex(chunks, use_dense="--dense" in sys.argv)
    index.save("data/processed/index.pkl")
    print(f"Indexed {len(docs)} docs / {len(chunks)} chunks -> data/processed/index.pkl")
