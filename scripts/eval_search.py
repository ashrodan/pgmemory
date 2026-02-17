"""Live evaluation of pgmemory search quality with real embeddings.

Seeds a test table with diverse memories, runs queries against ground truth,
and reports normalized metrics so providers are directly comparable.

Metrics:
    - Top-1 Accuracy: Is the best result correct?
    - MRR (Mean Reciprocal Rank): 1/rank of first correct result (higher = better)
    - P@3 (Precision at 3): Fraction of top-3 results that are relevant
    - Discrimination: gap between best relevant and best irrelevant similarity

Usage:
    uv run --extra openai --extra vertex --extra voyage python scripts/eval_search.py
    uv run --extra openai --extra vertex --extra voyage python scripts/eval_search.py --enrich
    uv run --extra openai --extra vertex --extra voyage python scripts/eval_search.py --provider gemini
    uv run --extra openai --extra vertex --extra voyage python scripts/eval_search.py --matrix
    uv run --extra openai --extra vertex --extra voyage python scripts/eval_search.py --threshold 0.15

Requires:
    - PGMEMORY_TEST_URL or local Postgres (postgresql+asyncpg://...)
    - OPENAI_API_KEY for OpenAI embeddings
    - GEMINI_API_KEY for Gemini embeddings
    - VOYAGE_API_KEY for Voyage (Anthropic) embeddings
"""
import argparse
import asyncio
import getpass
import os
import time

from pgmemory import Category, Memory, MemoryStore, SearchQuery
from pgmemory.embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    VertexEmbeddingProvider,
    VoyageEmbeddingProvider,
)


# ── Seed data ─────────────────────────────────────────────────────────

SEED_MEMORIES = [
    ("I love cake", Category.PREFERENCE),
    ("Prefers dark mode and compact layouts", Category.PREFERENCE),
    ("Favourite cuisine is Japanese", Category.PREFERENCE),
    ("Allergic to peanuts", Category.FACT),
    ("Works at Acme Corp as a data engineer", Category.FACT),
    ("Lives in Melbourne, Australia", Category.FACT),
    ("Born in 1990", Category.FACT),
    ("Currently building a RAG pipeline", Category.CONTEXT),
    ("Working on a deadline for Friday", Category.CONTEXT),
    ("Proficient in Python and SQL", Category.SKILL),
    ("Learning Rust", Category.SKILL),
    ("Always use UTC timestamps in APIs", Category.RULE),
    ("Never store passwords in plaintext", Category.RULE),
    ("Had a meeting with the design team yesterday", Category.EVENT),
]

# ── Ground truth: query → set of relevant memory texts ────────────────

GROUND_TRUTH: dict[str, set[str]] = {
    "im hungry": {
        "I love cake",
        "Favourite cuisine is Japanese",
        "Allergic to peanuts",
    },
    "what food does the user like": {
        "I love cake",
        "Favourite cuisine is Japanese",
        "Allergic to peanuts",
    },
    "UI preferences": {
        "Prefers dark mode and compact layouts",
    },
    "where does the user live": {
        "Lives in Melbourne, Australia",
    },
    "programming languages": {
        "Proficient in Python and SQL",
        "Learning Rust",
    },
    "what are the rules": {
        "Always use UTC timestamps in APIs",
        "Never store passwords in plaintext",
    },
    "what is the user working on": {
        "Currently building a RAG pipeline",
        "Working on a deadline for Friday",
    },
}

TEST_QUERIES = list(GROUND_TRUTH.keys())

PROVIDERS: dict[str, tuple[type[EmbeddingProvider], dict]] = {
    "openai": (OpenAIEmbeddingProvider, {"dimensionality": 1536}),
    "gemini": (VertexEmbeddingProvider, {"model": "gemini-embedding-001", "dimensionality": 768}),
    "voyage": (VoyageEmbeddingProvider, {"dimensionality": 1024}),
}


# ── Metrics ───────────────────────────────────────────────────────────

def compute_metrics(
    results: list,
    relevant: set[str],
) -> dict:
    """Compute normalized metrics for one query's results."""
    texts = [r.memory.text for r in results]
    sims = [r.similarity for r in results]

    # Top-1 accuracy
    top1_correct = texts[0] in relevant if texts else False

    # MRR: 1 / rank of first relevant result
    mrr = 0.0
    for i, t in enumerate(texts):
        if t in relevant:
            mrr = 1.0 / (i + 1)
            break

    # Precision@3
    top3_relevant = sum(1 for t in texts[:3] if t in relevant)
    p_at_3 = top3_relevant / min(3, len(texts)) if texts else 0.0

    # Discrimination: best relevant sim - best irrelevant sim
    best_relevant_sim = max(
        (s for t, s in zip(texts, sims) if t in relevant), default=0.0
    )
    best_irrelevant_sim = max(
        (s for t, s in zip(texts, sims) if t not in relevant), default=0.0
    )
    discrimination = best_relevant_sim - best_irrelevant_sim

    return {
        "top1_correct": top1_correct,
        "top1_text": texts[0] if texts else "-",
        "top1_sim": sims[0] if sims else 0.0,
        "mrr": mrr,
        "p_at_3": p_at_3,
        "discrimination": discrimination,
    }


# ── Core eval ─────────────────────────────────────────────────────────

def get_db_url() -> str:
    url = os.environ.get("PGMEMORY_TEST_URL")
    if url:
        return url
    user = os.environ.get("PGUSER", getpass.getuser())
    return f"postgresql+asyncpg://{user}@localhost:5432/pgmemory_test"


def make_embedder(provider_name: str) -> EmbeddingProvider:
    cls, kwargs = PROVIDERS[provider_name]
    return cls(**kwargs)


async def run_eval(
    provider_name: str,
    enrich: bool,
    threshold: float,
    *,
    percentile: float | None = None,
    collect: dict | None = None,
    verbose: bool = True,
):
    """Run one eval pass. If collect is provided, store per-query metrics."""
    db_url = get_db_url()
    embedder = make_embedder(provider_name)
    table_name = f"eval_{provider_name}_{'enrich' if enrich else 'plain'}"
    combo_key = f"{provider_name}+{'enrich' if enrich else 'plain'}"

    store = MemoryStore(
        db_url,
        embedder,
        table_name=table_name,
        enrich_embeddings=enrich,
    )

    try:
        await store.init()

        from sqlalchemy import text as sa_text
        async with store._session_factory() as db:
            await db.execute(sa_text(f"TRUNCATE TABLE {table_name}"))
            await db.commit()

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        memories = [
            Memory(
                app_name="eval",
                user_id="test_user",
                text=text,
                category=cat,
                importance=2,
                created_at=now,
                valid_from=now,
            )
            for text, cat in SEED_MEMORIES
        ]
        t0 = time.perf_counter()
        ids = await store.add_many(memories)
        seed_time = time.perf_counter() - t0
        if verbose:
            pct_info = f", percentile={percentile}" if percentile is not None else ""
            print(
                f"Seeded {len(ids)} memories (provider={provider_name}, enrich={enrich}{pct_info})"
                f" — embed+insert {seed_time:.3f}s ({seed_time/len(ids)*1000:.0f}ms/vec)\n"
            )

        query_times: list[float] = []
        for query_text in TEST_QUERIES:
            relevant = GROUND_TRUTH[query_text]

            # Always fetch with threshold=0 so we get full ranking
            t0 = time.perf_counter()
            results = await store.search(
                SearchQuery(
                    app_name="eval",
                    user_id="test_user",
                    text=query_text,
                    top_k=5,
                    similarity_threshold=0.0,
                    threshold_percentile=percentile,
                )
            )
            query_time = time.perf_counter() - t0
            query_times.append(query_time)

            metrics = compute_metrics(results, relevant)
            passes_threshold = results[0].combined_score >= threshold if results else False

            if verbose:
                print(f"{'─' * 70}")
                print(f"  QUERY: {query_text!r}  ({query_time*1000:.0f}ms)")
                print(f"  Expected: {relevant}")
                print(f"{'─' * 70}")
                for r in results:
                    is_rel = r.memory.text in relevant
                    tag = "HIT " if is_rel else "    "
                    print(
                        f"  [{tag}] sim={r.similarity:.4f} kw={r.keyword_score:.4f} "
                        f"rec={r.recency_score:.4f} combined={r.combined_score:.4f} "
                        f"| {r.memory.category.value}: {r.memory.text!r}"
                    )
                status = "PASS" if metrics["top1_correct"] else "MISS"
                print(
                    f"  >> {status} | MRR={metrics['mrr']:.2f} P@3={metrics['p_at_3']:.2f} "
                    f"disc={metrics['discrimination']:+.4f}"
                )
                if not passes_threshold:
                    print(f"  >> (top-1 combined {results[0].combined_score:.4f} < threshold {threshold})")
                print()

            metrics["query_time_ms"] = query_time * 1000
            if collect is not None:
                collect.setdefault(query_text, {})[combo_key] = metrics

        if verbose and query_times:
            avg_qt = sum(query_times) / len(query_times) * 1000
            print(
                f"  Timing: seed={seed_time:.3f}s, "
                f"avg query={avg_qt:.0f}ms, "
                f"total queries={sum(query_times):.3f}s\n"
            )

        if collect is not None:
            collect.setdefault("__timing__", {})[combo_key] = {
                "seed_time_s": seed_time,
                "seed_per_vec_ms": seed_time / len(ids) * 1000,
                "avg_query_ms": sum(query_times) / len(query_times) * 1000 if query_times else 0,
                "total_query_s": sum(query_times),
            }

        # Cleanup
        async with store._session_factory() as db:
            await db.execute(sa_text(f"TRUNCATE TABLE {table_name}"))
            await db.commit()
        if verbose:
            print("Cleaned up eval table.\n")

    finally:
        await store.close()


# ── Matrix display ────────────────────────────────────────────────────

def print_matrix(collect: dict, threshold: float):
    """Print normalized comparison matrix."""
    if not collect:
        return

    all_combos = set()
    for key, per_query in collect.items():
        if key == "__timing__":
            continue
        all_combos.update(per_query.keys())
    combos = sorted(all_combos)

    # ── Per-query breakdown ───────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  PER-QUERY RESULTS")
    print(f"{'=' * 90}\n")

    for query_text in TEST_QUERIES:
        per_query = collect.get(query_text, {})
        print(f"  {query_text!r}")
        print(f"  Expected: {GROUND_TRUTH[query_text]}")
        header = f"    {'Config':<25} {'Top-1?':>6} {'MRR':>5} {'P@3':>5} {'Disc':>7}  Top-1 hit"
        print(header)
        print(f"    {'─' * 85}")
        for combo in combos:
            m = per_query.get(combo)
            if m is None:
                print(f"    {combo:<25} {'(skipped)':>6}")
                continue
            mark = "Y" if m["top1_correct"] else "N"
            print(
                f"    {combo:<25} {mark:>6} {m['mrr']:>5.2f} {m['p_at_3']:>5.2f} "
                f"{m['discrimination']:>+7.4f}  {m['top1_text']!r}"
            )
        print()

    # ── Aggregate scores ──────────────────────────────────────────
    print(f"{'=' * 90}")
    print("  AGGREGATE METRICS")
    print(f"{'=' * 90}")
    print(f"  threshold={threshold}, {len(TEST_QUERIES)} queries, {len(SEED_MEMORIES)} memories\n")

    header = (
        f"  {'Config':<25} {'Top-1 Acc':>9} {'Avg MRR':>8} {'Avg P@3':>8} "
        f"{'Avg Disc':>9} {'Seed':>7} {'Avg Q':>7}"
    )
    print(header)
    print(f"  {'─' * 78}")

    timing_data = collect.get("__timing__", {})
    for combo in combos:
        query_metrics = [
            collect[q][combo]
            for q in TEST_QUERIES
            if combo in collect.get(q, {})
        ]
        if not query_metrics:
            print(f"  {combo:<25} {'(skipped)':>9}")
            continue

        n = len(query_metrics)
        top1_acc = sum(1 for m in query_metrics if m["top1_correct"]) / n
        avg_mrr = sum(m["mrr"] for m in query_metrics) / n
        avg_p3 = sum(m["p_at_3"] for m in query_metrics) / n
        avg_disc = sum(m["discrimination"] for m in query_metrics) / n

        t = timing_data.get(combo, {})
        seed_str = f"{t['seed_time_s']:.2f}s" if t else "-"
        query_str = f"{t['avg_query_ms']:.0f}ms" if t else "-"

        print(
            f"  {combo:<25} {top1_acc:>8.0%} {avg_mrr:>8.3f} {avg_p3:>8.3f} "
            f"{avg_disc:>+9.4f} {seed_str:>7} {query_str:>7}"
        )

    print()


async def run_matrix(threshold: float, percentile: float | None = None):
    """Run all provider x enrich combinations and print comparison."""
    collect: dict = {}
    for provider_name in PROVIDERS:
        for enrich in [False, True]:
            label = f"{provider_name}+{'enrich' if enrich else 'plain'}"
            print(f"\n{'=' * 70}")
            print(f"  {label}")
            print(f"{'=' * 70}\n")
            try:
                await run_eval(provider_name, enrich, threshold, percentile=percentile, collect=collect)
            except Exception as e:
                print(f"  SKIPPED: {e}\n")

    print_matrix(collect, threshold)


def main():
    parser = argparse.ArgumentParser(description="pgmemory search quality eval")
    parser.add_argument(
        "--provider", choices=list(PROVIDERS.keys()), default="openai",
        help="Embedding provider (default: openai)",
    )
    parser.add_argument("--enrich", action="store_true", help="Enable embedding enrichment")
    parser.add_argument("--matrix", action="store_true", help="Run all providers x enrich on/off")
    parser.add_argument("--threshold", type=float, default=0.2, help="Similarity threshold (default: 0.2)")
    parser.add_argument("--percentile", type=float, default=None, help="Percentile threshold 0.0-1.0 (e.g. 0.3 = filter bottom 30%%)")
    args = parser.parse_args()

    pct_info = f", percentile={args.percentile}" if args.percentile is not None else ""
    if args.matrix:
        print(f"\n{'=' * 70}")
        print(f"  pgmemory matrix eval — threshold={args.threshold}{pct_info}")
        print(f"{'=' * 70}\n")
        asyncio.run(run_matrix(threshold=args.threshold, percentile=args.percentile))
    else:
        print(f"\n{'=' * 70}")
        print(f"  pgmemory search eval — provider={args.provider}, enrich={args.enrich}, threshold={args.threshold}{pct_info}")
        print(f"{'=' * 70}\n")
        asyncio.run(run_eval(args.provider, enrich=args.enrich, threshold=args.threshold, percentile=args.percentile))


if __name__ == "__main__":
    main()
