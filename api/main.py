"""FastAPI service exposing the agent graph.

POST /ask        {"query": "..."} -> answer + citations + trace metadata
GET  /health     liveness probe (used by Docker healthcheck)
GET  /sources    what's indexed, for debuggability
"""
import sys
sys.path.insert(0, ".")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from src.agents.graph import answer as agent_answer
from src.index.store import HybridIndex

INDEX_PATH = "data/processed/index.pkl"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["index"] = HybridIndex.load(INDEX_PATH)
    yield
    state.clear()


app = FastAPI(title="HR Policy Copilot", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    route: str | None
    citations: list[str]
    revisions: int


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    return agent_answer(req.query, state["index"])


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(state["index"].chunks)}


@app.get("/sources")
def sources():
    seen = {}
    for c in state["index"].chunks:
        seen.setdefault(c.doc_id, {"title": c.title, "system": c.source_system,
                                   "region": c.region, "chunks": 0})
        seen[c.doc_id]["chunks"] += 1
    return seen
