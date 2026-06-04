# Built-in RAG test suite. Run: python -m project --test

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from .config import resolve_data_dir, settings
from .data_loader import load_documents, split_documents
from .pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str


def _check(name: str, ok: bool, detail: str) -> TestResult:
    return TestResult(name=name, passed=ok, detail=detail)


def _source_contains(result: dict[str, Any], keyword: str) -> bool:
    src = str(result.get("metadata", {}).get("source", "")).lower()
    return keyword.lower() in src


def run_test_suite(
    pipeline: RAGPipeline,
    *,
    run_llm: bool = False,
    reset_before: bool = False,
) -> list[TestResult]:
    """Run all checks and return results (does not raise)."""
    results: list[TestResult] = []

    try:
        data_dir = resolve_data_dir()
        results.append(_check("data_directory", True, str(data_dir)))
    except FileNotFoundError as exc:
        results.append(_check("data_directory", False, str(exc)))
        return results

    health = pipeline.health()
    results.append(
        _check(
            "vector_index_nonempty",
            health["chunks_indexed"] > 0,
            f"{health['chunks_indexed']} chunks indexed",
        )
    )
    results.append(
        _check(
            "embedding_model_config",
            health["embedding_model"] == settings.embedding_model_name,
            health["embedding_model"],
        )
    )

    if reset_before:
        try:
            added = pipeline.ingest(reset=True)
            results.append(
                _check("ingest_reset", added > 0, f"ingested {added} chunks")
            )
        except Exception as exc:
            results.append(_check("ingest_reset", False, str(exc)))

    try:
        raw = load_documents()
        results.append(
            _check("load_documents", len(raw) > 0, f"{len(raw)} raw document(s)")
        )
        chunks = split_documents(raw)
        results.append(
            _check(
                "split_documents",
                len(chunks) >= len(raw),
                f"{len(chunks)} chunks from {len(raw)} pages/files",
            )
        )
    except Exception as exc:
        results.append(_check("load_documents", False, str(exc)))
        results.append(_check("split_documents", False, "skipped"))

    if health["chunks_indexed"] == 0:
        results.append(
            _check(
                "retrieval_skipped",
                False,
                "No chunks in index. Run: python -m project --reset-index --ingest-only",
            )
        )
        return results

    retrieval_cases: list[tuple[str, str, float, Callable[[dict], bool]]] = [
        (
            "retrieve_transformer_paper",
            "What is attention is all you need",
            0.35,
            lambda r: _source_contains(r, "transformers.pdf"),
        ),
        (
            "retrieve_claude_paper",
            "What is claude used for",
            0.35,
            lambda r: _source_contains(r, "claude"),
        ),
    ]

    in_corpus_top_scores: list[float] = []

    for name, query, min_score, validator in retrieval_cases:
        try:
            hits = pipeline.retrieve(query, top_k=3, min_score=0.0)
            if not hits:
                results.append(_check(name, False, "no chunks returned"))
                continue
            top = hits[0]
            score_ok = top["similarity_score"] >= min_score
            content_ok = validator(top)
            ok = score_ok and content_ok
            in_corpus_top_scores.append(top["similarity_score"])
            results.append(
                _check(
                    name,
                    ok,
                    f"top_score={top['similarity_score']:.3f} "
                    f"source={top['metadata'].get('source', '?')}",
                )
            )
        except Exception as exc:
            results.append(_check(name, False, str(exc)))

    # Off-topic query should score lower than in-corpus questions
    try:
        ood_query = "How do I bake sourdough bread at home?"
        ood_hits = pipeline.retrieve(ood_query, top_k=3, min_score=0.0)
        if not ood_hits:
            results.append(
                _check(
                    "retrieve_out_of_corpus_low_score",
                    True,
                    "no chunks returned",
                )
            )
        elif in_corpus_top_scores:
            ood_score = ood_hits[0]["similarity_score"]
            best_in_corpus = max(in_corpus_top_scores)
            gap = best_in_corpus - ood_score
            ok = gap >= 0.1
            results.append(
                _check(
                    "retrieve_out_of_corpus_low_score",
                    ok,
                    f"ood_score={ood_score:.3f} best_in_corpus={best_in_corpus:.3f} "
                    f"gap={gap:.3f} (need gap >= 0.1)",
                )
            )
        else:
            results.append(
                _check(
                    "retrieve_out_of_corpus_low_score",
                    False,
                    "skipped (in-corpus checks did not run)",
                )
            )
    except Exception as exc:
        results.append(_check("retrieve_out_of_corpus_low_score", False, str(exc)))

    if run_llm:
        if not settings.groq_api_key:
            results.append(
                _check("rag_answer_groq", False, "GROQ_API_KEY not set in .env")
            )
        else:
            try:
                out = pipeline.ask(
                    "What is the Transformer architecture?",
                    top_k=3,
                    min_score=0.1,
                )
                answer = (out.get("answer") or "").lower()
                has_answer = len(answer) > 20 and "no relevant" not in answer
                grounded = any(
                    w in answer
                    for w in ("attention", "encoder", "decoder", "transformer")
                )
                results.append(
                    _check(
                        "rag_answer_groq",
                        has_answer and grounded,
                        f"confidence={out.get('confidence', 0):.3f}, "
                        f"preview={answer[:120]}...",
                    )
                )
            except Exception as exc:
                results.append(_check("rag_answer_groq", False, str(exc)))
    else:
        results.append(
            _check(
                "rag_answer_groq",
                True,
                "skipped (pass --test-llm to call Groq)",
            )
        )

    return results


def print_test_report(results: list[TestResult]) -> bool:
    """Print results; return True if all checks passed."""
    print("\n" + "=" * 60)
    print("RAG TEST REPORT")
    print("=" * 60)
    all_pass = True
    for r in results:
        icon = "PASS" if r.passed else "FAIL"
        if not r.passed:
            all_pass = False
        print(f"  [{icon}] {r.name}")
        print(f"         {r.detail}")
    print("=" * 60)
    passed = sum(1 for r in results if r.passed)
    print(f"  {passed}/{len(results)} checks passed")
    print("=" * 60)
    return all_pass
