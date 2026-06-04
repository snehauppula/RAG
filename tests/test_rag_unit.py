"""Fast unit tests (no Groq, no full embedding model download if cached)."""

from __future__ import annotations

from langchain_core.documents import Document

from project.data_loader import split_documents
from project.vector_store import VectorStoreManager


def test_split_documents_produces_chunks():
    docs = [
        Document(page_content="A" * 500, metadata={"source": "a.pdf", "page": 0}),
        Document(page_content="B" * 500, metadata={"source": "b.pdf", "page": 1}),
    ]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=50)
    assert len(chunks) >= 2


def test_stable_chunk_id_is_deterministic():
    doc = Document(
        page_content="Transformer attention mechanism",
        metadata={"source": "D:/data/pdf/Transformers.pdf", "page": 0},
    )
    id1 = VectorStoreManager._stable_chunk_id(doc, 0)
    id2 = VectorStoreManager._stable_chunk_id(doc, 0)
    assert id1 == id2
    assert len(id1) == 32


def test_similarity_formula():
    distance = 0.5
    similarity = 1.0 - (distance**2) / 2.0
    assert 0.8 < similarity < 0.9
