"""Document loading for the BonBon FAQ PDF.

Loads with ``pypdf`` directly (rather than a ``langchain-community`` loader)
to avoid pulling in ``langchain-community`` as a dependency.
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from .config import DATA_PATH


def _clean(text: str) -> str:
    """Collapse repeated whitespace while preserving paragraph breaks."""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: str | Path = DATA_PATH) -> list[Document]:
    """Load the PDF into one :class:`Document` per page.

    Each document carries ``source`` (file name) and ``page`` (1-indexed)
    metadata so answers and findings can cite where evidence came from.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input document not found: {path}")

    reader = PdfReader(str(path))
    documents: list[Document] = []
    for i, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "page": i},
            )
        )

    if not documents:
        raise ValueError(f"No extractable text found in {path}.")
    return documents
