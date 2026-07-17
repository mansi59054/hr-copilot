"""Multi-agent orchestration with LangGraph.

Graph:
    router -> retriever -> synthesizer -> critic -> (revise? -> synthesizer) -> END

Roles:
  router      classifies the query (policy_lookup | conflict_check | out_of_scope)
              and extracts region + topic, so retrieval is targeted.
  retriever   deterministic tool node: hybrid search + provenance packaging.
  synthesizer answers ONLY from retrieved chunks, cites chunk_ids, and must
              apply the precedence rule: newer dated announcements supersede
              older handbook text; regional handbooks beat global extracts.
  critic      LLM-as-a-judge gate: checks grounding (every claim traceable to a
              cited chunk) and conflict handling. One revision loop max -
              unbounded self-correction loops are a production anti-pattern.

State is a typed dict; every node is a pure function of state -> state delta,
which is what makes the graph testable node-by-node.
"""
from __future__ import annotations

import json
import os
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from src.index.store import HybridIndex

# ---------------------------------------------------------------- LLM client
def _llm(messages: list[dict], max_tokens: int = 1000) -> str:
    """Single place that talks to the model. Swap providers here."""
    from anthropic import Anthropic
    client = Anthropic()  # reads ANTHROPIC_API_KEY from env
    resp = client.messages.create(
        model=os.environ.get("MODEL_ID", "claude-sonnet-4-6"),
        max_tokens=max_tokens,
        messages=messages,
    )
    return resp.content[0].text


# ---------------------------------------------------------------- state
class AgentState(TypedDict, total=False):
    query: str
    route: str                 # policy_lookup | conflict_check | out_of_scope
    region: str | None
    retrieved: list[dict]      # chunk payloads with provenance
    draft: str
    critique: str
    verdict: str               # pass | revise | fail
    revisions: int
    answer: str


# ---------------------------------------------------------------- nodes
ROUTER_PROMPT = """You are the router for an internal HR policy assistant.
Classify the user query and extract targeting hints.

Query: {query}

Respond ONLY with JSON, no prose:
{{"route": "policy_lookup" | "conflict_check" | "out_of_scope",
  "region": "UK" | "India" | "Global" | null,
  "search_query": "<rewritten keyword-rich search query>"}}

out_of_scope = anything not answerable from HR/IT policy documents
conflict_check = the user is asking whether sources disagree or which rule wins"""


def router_node(state: AgentState) -> AgentState:
    raw = _llm([{"role": "user", "content": ROUTER_PROMPT.format(query=state["query"])}], 300)
    try:
        parsed = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        parsed = {"route": "policy_lookup", "region": None, "search_query": state["query"]}
    return {
        "route": parsed.get("route", "policy_lookup"),
        "region": parsed.get("region"),
        "query": state["query"],
        "retrieved": [],
        "revisions": 0,
        # stash rewritten query for the retriever
        "draft": parsed.get("search_query", state["query"]),
    }


def retriever_node(state: AgentState, index: HybridIndex, k: int = 6) -> AgentState:
    chunks = index.search(state.get("draft") or state["query"], k=k)
    # Soft region filter: keep matching region + region-agnostic, unless that empties the set
    region = state.get("region")
    if region:
        filtered = [c for c in chunks if c.region in (region, "Global", None)]
        chunks = filtered or chunks
    payload = [{
        "chunk_id": c.chunk_id, "text": c.text, "source": c.source_system,
        "title": c.title, "region": c.region, "timestamp": c.timestamp,
    } for c in chunks]
    return {"retrieved": payload}


SYNTH_PROMPT = """You are an HR policy assistant. Answer the query using ONLY the sources below.

Rules, in order:
1. Every factual claim must cite its source as [chunk_id].
2. Precedence: a dated Slack announcement from HR supersedes older handbook text on the same topic. Regional handbooks take precedence over global extracts for regional questions.
3. If sources conflict, say so explicitly, state which one wins under rule 2 and why, and mention the superseded source.
4. If the sources do not contain the answer, say exactly that. Do not guess.
5. Be concise. This is an internal tool, not an essay.

Query: {query}
Detected region: {region}

Sources:
{sources}"""


def synthesizer_node(state: AgentState) -> AgentState:
    sources = "\n\n".join(
        f"[{c['chunk_id']}] (system={c['source']}, region={c['region']}, ts={c['timestamp']})\n{c['text']}"
        for c in state["retrieved"]
    )
    extra = f"\n\nCritique of your previous draft (fix these issues):\n{state['critique']}" \
        if state.get("critique") else ""
    draft = _llm([{"role": "user", "content": SYNTH_PROMPT.format(
        query=state["query"], region=state.get("region"), sources=sources) + extra}])
    return {"draft": draft}


CRITIC_PROMPT = """You are a strict evaluator for an HR policy assistant. Judge the DRAFT against the SOURCES.

Checks:
1. Grounding: is every factual claim traceable to a cited [chunk_id] that actually supports it?
2. Conflict handling: if the sources disagree on the topic, does the draft surface the conflict and apply precedence (newer dated announcement > older handbook; regional > global)?
3. Scope: does the draft avoid inventing policy not present in sources?

Query: {query}

SOURCES:
{sources}

DRAFT:
{draft}

Respond ONLY with JSON:
{{"verdict": "pass" | "revise", "critique": "<specific, actionable issues, empty string if pass>"}}"""


def critic_node(state: AgentState) -> AgentState:
    sources = "\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in state["retrieved"])
    raw = _llm([{"role": "user", "content": CRITIC_PROMPT.format(
        query=state["query"], sources=sources, draft=state["draft"])}], 500)
    try:
        parsed = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        parsed = {"verdict": "pass", "critique": ""}
    verdict = parsed.get("verdict", "pass")
    if verdict == "revise" and state.get("revisions", 0) >= 1:
        verdict = "pass"  # hard cap: one revision loop
    return {
        "verdict": verdict,
        "critique": parsed.get("critique", ""),
        "revisions": state.get("revisions", 0) + (1 if verdict == "revise" else 0),
    }


def finalize_node(state: AgentState) -> AgentState:
    return {"answer": state["draft"]}


def out_of_scope_node(state: AgentState) -> AgentState:
    return {"answer": "This question is outside the scope of the indexed HR and IT "
                      "policy sources, so I can't answer it reliably. Try rephrasing, "
                      "or contact your HR Business Partner."}


# ---------------------------------------------------------------- graph
def build_graph(index: HybridIndex):
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("retriever", lambda s: retriever_node(s, index))
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("critic", critic_node)
    g.add_node("finalize", finalize_node)
    g.add_node("out_of_scope", out_of_scope_node)

    g.set_entry_point("router")
    g.add_conditional_edges("router", lambda s: s["route"],
                            {"policy_lookup": "retriever",
                             "conflict_check": "retriever",
                             "out_of_scope": "out_of_scope"})
    g.add_edge("retriever", "synthesizer")
    g.add_edge("synthesizer", "critic")
    g.add_conditional_edges("critic", lambda s: s["verdict"],
                            {"pass": "finalize", "revise": "synthesizer"})
    g.add_edge("finalize", END)
    g.add_edge("out_of_scope", END)
    return g.compile()


def answer(query: str, index: HybridIndex) -> dict:
    graph = build_graph(index)
    final = graph.invoke({"query": query})
    return {"answer": final.get("answer", ""),
            "route": final.get("route"),
            "citations": [c["chunk_id"] for c in final.get("retrieved", [])],
            "revisions": final.get("revisions", 0)}
