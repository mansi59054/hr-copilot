"""Golden evaluation dataset.

Deliberately balanced across the failure modes that matter for this system,
not just the happy path:
  - simple lookups (both regions)
  - CONFLICT cases where sources disagree and precedence must be applied
  - out-of-scope traps
  - paraphrase robustness (no keyword overlap with source text)
"""

GOLDEN = [
    {   # simple lookup, UK
        "query": "How many days of annual leave do UK employees get?",
        "expected_chunk_prefixes": ["pdf::uk_hr_policy_2024"],
        "must_mention": ["25"],
        "category": "simple_lookup",
    },
    {   # simple lookup, India
        "query": "What is the meal expense limit for business travel in India?",
        "expected_chunk_prefixes": ["pdf::india_hr_policy_2025"],
        "must_mention": ["2000"],
        "category": "simple_lookup",
    },
    {   # CONFLICT: Slack update supersedes UK handbook (2 vs 3 remote days)
        "query": "How many days per week can UK employees work remotely?",
        "expected_chunk_prefixes": ["slack::#hr-announcements::0", "pdf::uk_hr_policy_2024"],
        "must_mention": ["3"],
        "should_mention_conflict": True,
        "category": "conflict",
    },
    {   # CONFLICT: India probation changed for Engineering only
        "query": "What is the probation period for an engineering hire in India signing in April 2026?",
        "expected_chunk_prefixes": ["slack::#hr-announcements::4", "pdf::india_hr_policy_2025"],
        "must_mention": ["6"],
        "should_mention_conflict": True,
        "category": "conflict",
    },
    {   # CONFLICT: UK carry-over exception (one-time)
        "query": "Can UK employees carry unused 2025 leave into 2026?",
        "expected_chunk_prefixes": ["slack::#hr-announcements::1", "pdf::uk_hr_policy_2024"],
        "must_mention": ["5"],
        "should_mention_conflict": True,
        "category": "conflict",
    },
    {   # paraphrase, no keyword overlap ("WFH" never appears in sources)
        "query": "whats the wfh situation for the india office",
        "expected_chunk_prefixes": ["pdf::india_hr_policy_2025"],
        "must_mention": ["3 days"],
        "category": "paraphrase",
    },
    {   # cross-source security question
        "query": "I want to work from a cafe, what do I need to do about network security?",
        "expected_chunk_prefixes": ["pdf::global_security_policy"],
        "must_mention": ["VPN"],
        "category": "cross_source",
    },
    {   # out-of-scope trap
        "query": "What is Acme's stock price forecast for next quarter?",
        "expected_chunk_prefixes": [],
        "must_mention": [],
        "expect_out_of_scope": True,
        "category": "out_of_scope",
    },
]
