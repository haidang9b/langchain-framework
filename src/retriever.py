"""Retriever helpers and context formatting."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from .config import load_settings
from .vector_store import get_vectorstore


def get_retriever(k: int | None = None) -> VectorStoreRetriever:
    """Return a similarity retriever over the BonBon FAQ index."""
    s = load_settings()
    return get_vectorstore().as_retriever(search_kwargs={"k": k or s.retrieval_k})


def format_docs(docs: list[Document]) -> str:
    """Render retrieved documents into a single citable context string.

    Each chunk is tagged with its source file and page so downstream
    prompts (answer generation, groundedness validation) and the user can
    trace evidence back to the document.
    """

    if not docs:
        return "(no relevant context found)"

    blocks = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        blocks.append(f"[source: {source}, page: {page}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)
