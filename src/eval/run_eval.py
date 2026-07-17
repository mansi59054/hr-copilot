"""Two-layer evaluation:

Layer 1 (retrieval, deterministic, no LLM): hit-rate@k and MRR against
expected chunks. Runs in CI on every commit - fast and free.

Layer 2 (end-to-end, needs ANTHROPIC_API_KEY): runs the full graph, checks
must_mention strings, out-of-scope handling, and conflict surfacing.

Usage:
    python -m src.eval.run_eval             # layer 1 only
    python -m src.eval.run_eval --full      # both layers
"""
import sys
sys.path.insert(0, ".")

import json
import os

from src.eval.golden import GOLDEN
from src.index.store import HybridIndex


def eval_retrieval(index: HybridIndex, k: int = 6) -> dict:
    hits, rr_sum, n = 0, 0.0, 0
    per_case = []
    for case in GOLDEN:
        if not case["expected_chunk_prefixes"]:
            continue
        n += 1
        results = index.search(case["query"], k=k)
        ids = [c.chunk_id for c in results]
        found_rank = None
        for rank, cid in enumerate(ids):
            if any(cid.startswith(p) for p in case["expected_chunk_prefixes"]):
                found_rank = rank
                break
        hit = found_rank is not None
        hits += int(hit)
        rr_sum += (1.0 / (found_rank + 1)) if hit else 0.0
        per_case.append({"query": case["query"], "hit": hit, "rank": found_rank})
    return {"hit_rate": hits / n, "mrr": rr_sum / n, "n": n, "cases": per_case}


def eval_end_to_end(index: HybridIndex) -> dict:
    from src.agents.graph import answer
    results = []
    for case in GOLDEN:
        out = answer(case["query"], index)
        text = out["answer"].lower()
        normalized = text.replace(",", "")
        ok_mentions = all(m.lower() in normalized for m in case["must_mention"])
        ok_scope = ("outside the scope" in text) == case.get("expect_out_of_scope", False)
        ok_conflict = True
        if case.get("should_mention_conflict"):
            ok_conflict = any(w in text for w in ["supersede", "conflict", "updated", "exception", "changed", "however"])
        results.append({
            "query": case["query"], "category": case["category"],
            "mentions_ok": ok_mentions, "scope_ok": ok_scope,
            "conflict_ok": ok_conflict, "revisions": out["revisions"],
            "passed": ok_mentions and ok_scope and ok_conflict,
        })
    passed = sum(r["passed"] for r in results)
    return {"pass_rate": passed / len(results), "n": len(results), "cases": results}


if __name__ == "__main__":
    index = HybridIndex.load("data/processed/index.pkl")
    retrieval = eval_retrieval(index)
    print(json.dumps({"retrieval": {k: v for k, v in retrieval.items() if k != "cases"}}, indent=2))
    for c in retrieval["cases"]:
        print(f"  {'HIT ' if c['hit'] else 'MISS'} rank={c['rank']} :: {c['query']}")

    if "--full" in sys.argv:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY for full eval")
        e2e = eval_end_to_end(index)
        print(json.dumps({"end_to_end": {k: v for k, v in e2e.items() if k != "cases"}}, indent=2))
        for c in e2e["cases"]:
            flag = "PASS" if c["passed"] else "FAIL"
            print(f"  {flag} [{c['category']}] rev={c['revisions']} :: {c['query']}")
