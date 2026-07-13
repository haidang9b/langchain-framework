"""Chunking strategy with configurable size and overlap."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import load_settings


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split page-level documents into overlapping chunks for embedding.

    ``RecursiveCharacterTextSplitter`` tries paragraph, then line, then word,
    then character boundaries — a sensible default for prose FAQ content.
    Defaults come from :class:`Settings` but can be overridden per call.
    """

    s = load_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or s.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else s.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    return splitter.split_documents(documents)
