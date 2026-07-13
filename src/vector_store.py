"""Chroma vector store: build, persist, and load.

The store is persisted to ``.chroma/`` (git-ignored). ``get_vectorstore``
is idempotent — it loads the existing index if present, or builds it once
from the PDF — so demos and notebooks stay fast on re-run.
"""

from __future__ import annotations

from langchain_chroma import Chroma

from .config import CHROMA_DIR, COLLECTION_NAME
from .llm import get_embeddings
from .loaders import load_pdf
from .splitters import split_documents


def _has_persisted_index() -> bool:
    """True if a non-empty Chroma directory already exists on disk."""
    return CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())


def build_vectorstore(force: bool = False) -> Chroma:
    """Load the PDF, chunk it, embed the chunks, and persist to Chroma.

    Args:
        force: rebuild even if a persisted index already exists.
    """

    if _has_persisted_index() and not force:
        return get_vectorstore()

    documents = load_pdf()
    chunks = split_documents(documents)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    return store


def get_vectorstore() -> Chroma:
    """Return the persisted store, building it first if it does not exist."""

    if not _has_persisted_index():
        return build_vectorstore(force=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
