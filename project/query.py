"""
Retrieval and generation: retriever, LLM, and RAG chains.

Notebook source: cells 12, 20, 22 (RAGRetriever, GroqLLM, rag_advanced, AdvancedRAGPipeline).
Pipeline role: query -> relevant chunks -> LLM answer with optional sources/citations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from .config import settings
from .embedding import EmbeddingManager
from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Query-based retrieval from Chroma (notebook cell 12)."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        embedding_manager: EmbeddingManager,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve ranked chunks for a query.

        Uses the same similarity formula as the notebook:
        similarity = 1.0 - (distance**2) / 2.0 for unit-normalized embeddings.
        """
        top_k = top_k if top_k is not None else settings.default_top_k
        score_threshold = (
            score_threshold
            if score_threshold is not None
            else settings.default_score_threshold
        )

        logger.info(
            "Retrieving for query=%r top_k=%s threshold=%s",
            query,
            top_k,
            score_threshold,
        )

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        retrieved_docs: list[dict[str, Any]] = []

        try:
            if self.vector_store.collection is None:
                raise ValueError("Vector store collection is not initialized.")

            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
            )

            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    similarity_score = 1.0 - (distance**2) / 2.0
                    if similarity_score >= score_threshold:
                        retrieved_docs.append(
                            {
                                "id": doc_id,
                                "content": document,
                                "metadata": metadata,
                                "similarity_score": similarity_score,
                                "distance": distance,
                                "rank": i + 1,
                            }
                        )

                logger.info(
                    "Retrieved %s document(s) after filtering",
                    len(retrieved_docs),
                )
            else:
                logger.info("No documents found for query")

            return retrieved_docs

        except Exception as exc:
            logger.error("Error during retrieval: %s", exc)
            return []


class GroqLLM:
    """Groq chat model wrapper (notebook cell 20)."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.groq_model_name
        self.api_key = api_key or settings.require_groq_api_key()

        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=settings.groq_temperature,
            max_tokens=settings.groq_max_tokens,
        )
        logger.info("Initialized Groq LLM with model: %s", self.model_name)

    def generate_response(self, query: str, context: str, max_length: int = 500) -> str:
        """Structured RAG prompt (notebook: generate_response)."""
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer: Provide a clear and informative answer based on the context above. If the context doesn't contain enough information to answer the question, say so.""",
        )
        formatted_prompt = prompt_template.format(context=context, question=query)

        try:
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages)
            return response.content or ""
        except Exception as exc:
            return f"Error generating response: {exc}"

    def generate_response_simple(self, query: str, context: str) -> str:
        """Simple RAG prompt (notebook: generate_response_simple)."""
        simple_prompt = f"""Based on this context: {context}

Question: {query}

Answer:"""

        try:
            messages = [HumanMessage(content=simple_prompt)]
            response = self.llm.invoke(messages)
            return response.content or ""
        except Exception as exc:
            return f"Error: {exc}"


# --- Advanced RAG (notebook cell 22) ---


def _doc_source(meta: dict[str, Any]) -> str:
    raw = meta.get("source_file") or meta.get("source", "unknown")
    return Path(raw).name if raw != "unknown" else raw


def _doc_preview(text: str, max_len: int = 300) -> str:
    return text if len(text) <= max_len else text[:max_len] + "..."


def _build_sources(
    results: list[dict[str, Any]],
    preview_len: int = 300,
) -> list[dict[str, Any]]:
    return [
        {
            "source": _doc_source(doc["metadata"]),
            "page": doc["metadata"].get(
                "page", doc["metadata"].get("page_label", "unknown")
            ),
            "score": doc["similarity_score"],
            "preview": _doc_preview(doc["content"], preview_len),
        }
        for doc in results
    ]


def _invoke_llm(groq_llm: GroqLLM, prompt: str) -> str:
    response = groq_llm.llm.invoke([HumanMessage(content=prompt)])
    return response.content or ""


def _stream_llm(groq_llm: GroqLLM, prompt: str) -> str:
    logger.info("Streaming answer...")
    parts: list[str] = []
    for chunk in groq_llm.llm.stream([HumanMessage(content=prompt)]):
        text = chunk.content or ""
        if text:
            print(text, end="", flush=True)
            parts.append(text)
    print()
    return "".join(parts)


def rag_advanced(
    query: str,
    retriever: RAGRetriever,
    llm: GroqLLM,
    top_k: int = 5,
    min_score: float = 0.2,
    return_context: bool = False,
) -> dict[str, Any]:
    """RAG with answer, sources, confidence, optional context (notebook cell 22)."""
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        out: dict[str, Any] = {
            "answer": "No relevant context found.",
            "sources": [],
            "confidence": 0.0,
        }
        if return_context:
            out["context"] = ""
        return out

    context = "\n\n".join(doc["content"] for doc in results)
    sources = _build_sources(results)
    confidence = max(doc["similarity_score"] for doc in results)

    prompt = (
        "Use the following context to answer the question concisely.\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    answer = _invoke_llm(llm, prompt)

    output: dict[str, Any] = {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
    }
    if return_context:
        output["context"] = context
    return output


class AdvancedRAGPipeline:
    """Citations, streaming, summarization, history (notebook cell 22)."""

    def __init__(self, retriever: RAGRetriever, llm: GroqLLM) -> None:
        self.retriever = retriever
        self.llm = llm
        self.history: list[dict[str, Any]] = []

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.2,
        stream: bool = False,
        summarize: bool = False,
    ) -> dict[str, Any]:
        results = self.retriever.retrieve(
            question, top_k=top_k, score_threshold=min_score
        )
        if not results:
            answer = "No relevant context found."
            sources: list[dict[str, Any]] = []
        else:
            context = "\n\n".join(doc["content"] for doc in results)
            sources = _build_sources(results, preview_len=120)
            prompt = (
                "Use the following context to answer the question concisely.\n"
                f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )
            answer = (
                _stream_llm(self.llm, prompt)
                if stream
                else _invoke_llm(self.llm, prompt)
            )

        citations = [
            f"[{i + 1}] {src['source']} (page {src['page']})"
            for i, src in enumerate(sources)
        ]
        answer_with_citations = (
            answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer
        )

        summary = None
        if summarize and answer and answer != "No relevant context found.":
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary = _invoke_llm(self.llm, summary_prompt)

        self.history.append(
            {
                "question": question,
                "answer": answer,
                "sources": sources,
                "summary": summary,
            }
        )

        return {
            "question": question,
            "answer": answer_with_citations,
            "sources": sources,
            "summary": summary,
            "history": self.history,
        }
