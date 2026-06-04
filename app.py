"""
Streamlit UI for the RAG pipeline (showcase / demo).

Run from repo root:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from project import RAGPipeline
from project.config import settings


@st.cache_resource(show_spinner="Loading embedding model and vector store...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline.create()


def confidence_label(score: float) -> tuple[str, str]:
    if score >= 0.4:
        return "strong", "success"
    if score >= 0.15:
        return "weak", "warning"
    return "poor", "error"


def main() -> None:
    st.set_page_config(
        page_title="RAG PDF Assistant",
        page_icon="📚",
        layout="wide",
    )

    st.title("RAG PDF Assistant")
    st.caption(
        "Ask questions about your PDFs. Answers are generated from retrieved "
        "chunks using Groq + Chroma + Sentence Transformers."
    )

    try:
        pipeline = get_pipeline()
    except Exception as exc:
        st.error(f"Failed to start pipeline: {exc}")
        st.stop()

    health = pipeline.health()

    with st.sidebar:
        st.header("System status")
        st.metric("Chunks indexed", health["chunks_indexed"])
        st.write(f"**Data:** `{health['data_dir']}`")
        st.write(f"**Embeddings:** `{health['embedding_model']}`")
        st.write(f"**LLM:** `{health['llm_model']}`")

        if health["groq_configured"]:
            st.success("Groq API key loaded")
        else:
            st.error("GROQ_API_KEY missing in `.env`")

        st.divider()
        st.header("Index")
        if st.button("Rebuild index from PDFs", type="primary", use_container_width=True):
            with st.spinner("Loading, chunking, embedding..."):
                try:
                    added = pipeline.ingest(reset=True)
                    st.success(f"Indexed {added} chunks.")
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        st.divider()
        st.header("Retrieval settings")
        top_k = st.slider("Top K chunks", 1, 10, 5)
        min_score = st.slider("Min similarity", 0.0, 1.0, 0.1, 0.05)

    question = st.text_area(
        "Your question",
        placeholder="e.g. What is attention is all you need?",
        height=100,
    )

    col_ask, col_retrieve = st.columns(2)
    ask_clicked = col_ask.button("Ask (RAG answer)", type="primary", use_container_width=True)
    retrieve_clicked = col_retrieve.button(
        "Preview chunks only", use_container_width=True
    )

    if not question.strip():
        st.info("Enter a question above to get started.")
        return

    if retrieve_clicked:
        with st.spinner("Retrieving..."):
            hits = pipeline.retrieve(question, top_k=top_k, min_score=min_score)
        st.subheader("Retrieved chunks")
        if not hits:
            st.warning("No chunks matched your filters.")
            return
        for hit in hits:
            src = hit["metadata"].get("source", "unknown")
            page = hit["metadata"].get("page", hit["metadata"].get("page_label", "?"))
            with st.expander(
                f"Rank {hit['rank']} | score {hit['similarity_score']:.3f} | "
                f"{src} (p. {page})"
            ):
                st.write(hit["content"])
        return

    if ask_clicked:
        if not health["groq_configured"]:
            st.error(
                "Set `GROQ_API_KEY` in `d:\\RAG\\.env` then refresh this page."
            )
            return

        with st.spinner("Retrieving context and generating answer..."):
            try:
                result = pipeline.ask(
                    question,
                    top_k=top_k,
                    min_score=min_score,
                    return_context=True,
                )
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                return

        st.subheader("Answer")
        st.markdown(result.get("answer", "_No answer_"))

        if "confidence" in result:
            label, level = confidence_label(float(result["confidence"]))
            st.metric("Retrieval confidence", f"{result['confidence']:.3f} ({label})")

        sources = result.get("sources") or []
        if sources:
            st.subheader("Sources")
            for i, src in enumerate(sources, 1):
                st.markdown(
                    f"**[{i}]** `{src['source']}` — page {src['page']} "
                    f"(score {src['score']:.3f})"
                )
                with st.expander("Preview"):
                    st.write(src.get("preview", ""))

        context = result.get("context")
        if context:
            with st.expander("Full retrieved context"):
                st.text(context[:8000])


if __name__ == "__main__":
    main()
