"""
RAG pipeline package.

Quick start::

    from project import RAGPipeline

    rag = RAGPipeline.create()
    rag.ingest(reset=True)
    result = rag.ask("What is attention?")
    rag.print_answer(result)

Dependency graph::

    config -> data_loader, embedding, vector_store -> query -> pipeline -> main
"""

from .config import (
    Settings,
    resolve_data_dir,
    resolve_vector_store_dir,
    settings,
)
from .data_loader import load_documents, load_pdf_files, split_documents
from .embedding import EmbeddingManager
from .pipeline import RAGPipeline
from .query import (
    AdvancedRAGPipeline,
    GroqLLM,
    RAGRetriever,
    rag_advanced,
)
from .vector_store import VectorStoreManager

__all__ = [
    "Settings",
    "settings",
    "resolve_data_dir",
    "resolve_vector_store_dir",
    "load_documents",
    "load_pdf_files",
    "split_documents",
    "EmbeddingManager",
    "VectorStoreManager",
    "RAGRetriever",
    "GroqLLM",
    "rag_advanced",
    "AdvancedRAGPipeline",
    "RAGPipeline",
]
