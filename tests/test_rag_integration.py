"""Integration tests (loads embedding model; needs data/pdf and vector index)."""

from __future__ import annotations

import pytest

from project import RAGPipeline
from project.config import resolve_data_dir


@pytest.fixture(scope="module")
def pipeline() -> RAGPipeline:
    return RAGPipeline.create()


def test_data_dir_exists():
    assert resolve_data_dir().is_dir()


def test_health_has_chunks(pipeline: RAGPipeline):
    health = pipeline.health()
    assert health["chunks_indexed"] > 0, (
        "Run: python -m project --reset-index --ingest-only"
    )


def test_retrieve_transformer_query(pipeline: RAGPipeline):
    hits = pipeline.retrieve("What is attention is all you need", top_k=3)
    assert len(hits) >= 1
    assert hits[0]["similarity_score"] > 0.3
    source = hits[0]["metadata"].get("source", "").lower()
    assert "transformers" in source
