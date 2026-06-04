"""
CLI entry point for the RAG pipeline.

Run from repo root:
  python -m project --ask "What is the Transformer?"
  python -m project --reset-index
  python -m project --retrieve-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .pipeline import RAGPipeline
from .testing import print_test_report, run_test_suite

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _print_health(pipeline: RAGPipeline) -> None:
    print("RAG Pipeline")
    print("-" * 40)
    for key, value in pipeline.health().items():
        print(f"  {key}: {value}")


def run_retrieval_demo(pipeline: RAGPipeline) -> None:
    for q in [
        "What is attention is all you need",
        "What is claude used for",
        "Unified Multi-task Learning Framework",
    ]:
        print("\n" + "=" * 60)
        print("Query:", q)
        results = pipeline.retrieve(q, top_k=3, min_score=0.0)
        print(f"Retrieved {len(results)} chunk(s)")
        if results:
            top = results[0]
            print(f"  Top score: {top['similarity_score']:.4f}")
            print(f"  Preview: {top['content'][:200]}...")


def run_builtin_demos(pipeline: RAGPipeline) -> None:
    """Notebook-style rag_advanced + AdvancedRAGPipeline examples."""
    result = pipeline.ask(
        "Hard Negative Mining Techniques",
        top_k=3,
        min_score=0.1,
        return_context=True,
    )
    pipeline.print_answer(result)
    if result.get("context"):
        print("\nContext preview:", result["context"][:300])

    print("\n" + "=" * 60)
    print("Demo: AdvancedRAGPipeline (stream + summarize)")
    adv = pipeline.ask(
        "what is attention is all you need",
        top_k=3,
        min_score=0.1,
        stream=True,
        summarize=True,
        advanced=True,
    )
    pipeline.print_answer(adv)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG: ingest documents, retrieve chunks, answer with Groq."
    )
    parser.add_argument(
        "--ask",
        type=str,
        metavar="QUESTION",
        help="Ask one question (full RAG: retrieve + generate).",
    )
    parser.add_argument(
        "--reset-index",
        action="store_true",
        help="Wipe the vector index and re-ingest from data/pdf (recommended vs --reindex).",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Deprecated alias for --reset-index.",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only ingest; skip query demos.",
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Only retrieval demos (no Groq API key needed).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.1,
        help="Minimum similarity score for chunks (default: 0.1).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="With --ask, print raw JSON instead of formatted text.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run built-in RAG test suite (retrieval; optional LLM with --test-llm).",
    )
    parser.add_argument(
        "--test-llm",
        action="store_true",
        help="With --test, also call Groq for a full RAG answer.",
    )
    args = parser.parse_args()

    reset = args.reset_index or args.reindex
    pipeline = RAGPipeline.create()
    _print_health(pipeline)

    if args.test:
        results = run_test_suite(
            pipeline,
            run_llm=args.test_llm,
            reset_before=reset,
        )
        ok = print_test_report(results)
        sys.exit(0 if ok else 1)

    added = pipeline.ingest(reset=reset)
    if added:
        print(f"\nIngested {added} chunk(s). Total: {pipeline.chunk_count}")

    if args.ask:
        try:
            result = pipeline.ask(
                args.ask,
                top_k=args.top_k,
                min_score=args.min_score,
                return_context=True,
            )
            if args.json:
                print(json.dumps(result, indent=2, default=str))
            else:
                pipeline.print_answer(result)
        except ValueError as exc:
            logger.error("%s", exc)
            sys.exit(1)
        return

    if not args.ingest_only:
        run_retrieval_demo(pipeline)

    if args.ingest_only or args.retrieve_only:
        return

    try:
        run_builtin_demos(pipeline)
    except ValueError as exc:
        logger.error("%s", exc)
        logger.error("Set GROQ_API_KEY in .env. Use --retrieve-only to skip the LLM.")
        sys.exit(1)


if __name__ == "__main__":
    main()
