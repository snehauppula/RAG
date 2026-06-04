"""
Central configuration for the RAG pipeline.

Notebook source: cells 2, 4, 6, 8, 12, 18, 20 (paths and hyperparameters scattered there).
Why here: one place for paths, model names, and secrets so other modules stay testable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root and/or project/ (supports project/.env)
_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
for _env_file in (_REPO_ROOT / ".env", _PKG_DIR / ".env"):
    if _env_file.is_file():
        load_dotenv(_env_file)


def _repo_root() -> Path:
    """Project lives under repo root; config.py is in project/."""
    return Path(__file__).resolve().parent.parent


def resolve_data_dir() -> Path:
    """
    Resolve document folder (notebook: pdf_data_dir in cell 2).

    Works when cwd is repo root or project/ or notebook/.
    """
    candidates = (
        _repo_root() / "data" / "pdf",
        Path("data/pdf"),
        Path("../data/pdf"),
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "No data directory found. Expected data/pdf at repo root "
        "(or data/pdf relative to current working directory)."
    )


def resolve_vector_store_dir() -> Path:
    """Resolve Chroma persist path (notebook: vector_store_dir in cell 8)."""
    if (_repo_root() / "data" / "pdf").is_dir():
        path = _repo_root() / "data" / "vector_store"
    elif Path("data/pdf").is_dir():
        path = Path("data/vector_store")
    else:
        path = Path("../data/vector_store")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    """Immutable settings used across the RAG pipeline."""

    # Document ingestion (cell 4)
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Embeddings (cell 6)
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Vector store (cell 8)
    collection_name: str = "pdf_documents"

    # Retrieval defaults (cell 12)
    default_top_k: int = 5
    default_score_threshold: float = 0.0

    # LLM (cell 20)
    groq_model_name: str = "llama-3.1-8b-instant"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 1024

    @property
    def groq_api_key(self) -> str | None:
        return os.environ.get("GROQ_API_KEY")

    def require_groq_api_key(self) -> str:
        key = self.groq_api_key
        if not key:
            raise ValueError(
                "Groq API key is required. Set GROQ_API_KEY in .env or the environment."
            )
        return key


settings = Settings()
