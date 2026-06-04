"""
ChromaDB vector store: persistence only, no embedding logic.

Notebook source: cell 8 (VectorStoreManager, vector_store_dir).
Pipeline role: store chunk text + embeddings + metadata for later retrieval.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from langchain_core.documents import Document

from .config import resolve_vector_store_dir, settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Persistent Chroma collection for document chunks (notebook: VectorStoreManager)."""

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: str | Path | None = None,
    ) -> None:
        self.collection_name = collection_name or settings.collection_name
        self.persist_directory = str(
            persist_directory or resolve_vector_store_dir()
        )
        self.client: chromadb.PersistentClient | None = None
        self.collection: Any = None
        self._initialize_vectorstore()

    def _initialize_vectorstore(self) -> None:
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Vector store for PDF documents",
                    "hnsw:space": "cosine",
                },
            )
            logger.info(
                "Collection initialized: %s (%s documents)",
                self.collection_name,
                self.collection.count(),
            )
        except Exception as exc:
            logger.error("Error initializing vector store: %s", exc)
            raise

    def get_document_count(self) -> int:
        """Return number of stored chunks."""
        if self.collection is None:
            return 0
        return int(self.collection.count())

    def reset_collection(self) -> int:
        """
        Delete and recreate the collection (clean re-ingest, no duplicates).

        Returns the number of chunks removed.
        """
        if self.client is None:
            raise ValueError("Vector store not initialized.")
        removed = self.get_document_count()
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._initialize_vectorstore()
        return removed

    @staticmethod
    def _stable_chunk_id(doc: Document, index: int) -> str:
        """Same source+page+text -> same id (safe upsert on re-ingest)."""
        source = str(doc.metadata.get("source", ""))
        page = str(doc.metadata.get("page", doc.metadata.get("page_label", "")))
        text_key = doc.page_content[:500]
        payload = f"{source}|{page}|{text_key}|{index}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def add_documents(
        self,
        documents: list[Document],
        embeddings: np.ndarray,
    ) -> int:
        """
        Add chunked documents and precomputed embeddings (notebook: add_documents).

        Inputs: LangChain Documents + matching embedding matrix.
        Outputs: number of documents added.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have the same length.")
        if self.collection is None:
            raise ValueError("Vector store not initialized.")

        logger.info("Adding %s documents to vector store", len(documents))

        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        documents_list: list[str] = []
        embeddings_list: list[list[float]] = []

        for i, doc in enumerate(documents):
            ids.append(self._stable_chunk_id(doc, i))
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)
            documents_list.append(doc.page_content)
            embeddings_list.append(embeddings[i].tolist())

        self.collection.add(
            ids=ids,
            documents=documents_list,
            embeddings=embeddings_list,
            metadatas=metadatas,
        )

        count = self.get_document_count()
        logger.info(
            "Successfully added %s documents. Total in store: %s",
            len(documents),
            count,
        )
        return len(documents)
