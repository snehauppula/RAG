"""
High-level RAG pipeline: one object wires ingest, retrieve, and answer.

Why this module exists:
- Beginners interact with one class instead of six modules.
- main.py and notebooks call the same API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .config import resolve_data_dir, settings
from .data_loader import load_documents, split_documents
from .embedding import EmbeddingManager
from .query import AdvancedRAGPipeline, GroqLLM, RAGRetriever, rag_advanced
from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class RAGPipeline:
    """
    End-to-end RAG system.

    Components:
        embedder   -> turns text into vectors
        vectorstore -> saves chunks in ChromaDB
        retriever  -> finds relevant chunks for a question
    """

    embedder: EmbeddingManager
    vectorstore: VectorStoreManager
    retriever: RAGRetriever

    @classmethod
    def create(cls) -> RAGPipeline:
        """Build a ready-to-use pipeline (loads embedding model once)."""
        embedder = EmbeddingManager()
        vectorstore = VectorStoreManager()
        retriever = RAGRetriever(vectorstore, embedder)
        return cls(embedder=embedder, vectorstore=vectorstore, retriever=retriever)

    @property
    def chunk_count(self) -> int:
        return self.vectorstore.get_document_count()

    def health(self) -> dict[str, Any]:
        """Quick status for debugging and demos."""
        return {
            "data_dir": str(resolve_data_dir()),
            "embedding_model": settings.embedding_model_name,
            "llm_model": settings.groq_model_name,
            "chunks_indexed": self.chunk_count,
            "groq_configured": bool(settings.groq_api_key),
        }

    def ingest(self, reset: bool = False) -> int:
        """
        Load PDFs/TXT, chunk, embed, and store.

        reset=False: skip if the index already has chunks.
        reset=True:  wipe the collection and rebuild (no duplicate chunks).
        """
        if reset:
            removed = self.vectorstore.reset_collection()
            logger.info("Reset vector index (removed %s old chunks).", removed)
        elif self.chunk_count > 0:
            logger.info(
                "Index already has %s chunks. Skipping ingest. "
                "Use reset=True or --reset-index to rebuild.",
                self.chunk_count,
            )
            return 0

        logger.info("Loading documents from %s", resolve_data_dir())
        raw_docs = load_documents()
        split_docs = split_documents(raw_docs)
        texts = [doc.page_content for doc in split_docs]
        embeddings = self.embedder.generate_embeddings(texts)
        return self.vectorstore.add_documents(split_docs, embeddings)

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return ranked chunks only (no LLM)."""
        return self.retriever.retrieve(
            question,
            top_k=top_k or settings.default_top_k,
            score_threshold=min_score
            if min_score is not None
            else settings.default_score_threshold,
        )

    def ask(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.1,
        return_context: bool = False,
        stream: bool = False,
        summarize: bool = False,
        advanced: bool = False,
    ) -> dict[str, Any]:
        """
        Full RAG: retrieve context, then generate an answer with Groq.

        advanced=False -> rag_advanced (answer + sources + confidence)
        advanced=True  -> AdvancedRAGPipeline (citations, optional stream/summary)
        """
        llm = GroqLLM()
        if advanced:
            pipeline = AdvancedRAGPipeline(self.retriever, llm)
            return pipeline.query(
                question,
                top_k=top_k,
                min_score=min_score,
                stream=stream,
                summarize=summarize,
            )
        return rag_advanced(
            question,
            self.retriever,
            llm,
            top_k=top_k,
            min_score=min_score,
            return_context=return_context,
        )

    def print_answer(self, result: dict[str, Any]) -> None:
        """Pretty-print a result from ask() or rag_advanced()."""
        answer = result.get("answer", "")
        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer)

        if "confidence" in result:
            quality = (
                "strong"
                if result["confidence"] >= 0.4
                else "weak"
                if result["confidence"] >= 0.15
                else "poor"
            )
            print(f"\nRetrieval confidence: {result['confidence']:.3f} ({quality})")
            if quality == "poor":
                print(
                    "Tip: try rephrasing the question or run ingest with reset=True."
                )

        sources = result.get("sources") or []
        if sources:
            print("\nSOURCES")
            print("-" * 60)
            for i, src in enumerate(sources, 1):
                print(
                    f"  [{i}] {src['source']} (page {src['page']}) "
                    f"score={src['score']:.3f}"
                )

        if result.get("summary"):
            print("\nSUMMARY")
            print("-" * 60)
            print(result["summary"])
