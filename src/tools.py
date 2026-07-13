"""LangChain tools the agent can call.

Tools are defined with the ``@tool`` decorator so the model sees their name,
typed signature, and docstring as the tool schema. ``search_document`` is the
primary tool: it runs retrieval over the BonBon FAQ.
"""

from __future__ import annotations

from langchain_core.tools import tool

from .retriever import format_docs, get_retriever


@tool
def search_document(query: str) -> str:
    """Search the BonBon IT Service Desk FAQ for content relevant to the query.

    Use this for any question about IT support topics (passwords, Wi-Fi,
    software installation, malware, shared drives, etc.). Returns the most
    relevant chunks with their source and page.
    """
    docs = get_retriever().invoke(query)
    return format_docs(docs)


@tool
def get_source_context(query: str, k: int = 3) -> str:
    """Return raw source chunks (with page numbers) for citing evidence.

    Use this when you need to quote or cite exact FAQ passages that back up
    an answer or finding.
    """
    docs = get_retriever(k=k).invoke(query)
    return format_docs(docs)


@tool
def validate_question(question: str) -> str:
    """Quick guard: judge whether a question is answerable from an IT FAQ.

    Returns a short note. Purely heuristic — the agent should still retrieve
    before concluding a topic is out of scope.
    """
    it_terms = (
        "password",
        "wifi",
        "wi-fi",
        "network",
        "internet",
        "software",
        "install",
        "malware",
        "virus",
        "email",
        "drive",
        "account",
        "computer",
        "slow",
        "file",
        "security",
        "reset",
    )
    lowered = question.lower()
    if any(term in lowered for term in it_terms):
        return "In scope: this looks like an IT support question the FAQ may cover."
    return (
        "Possibly out of scope: no obvious IT support keywords detected. "
        "Retrieve anyway before deciding."
    )


ALL_TOOLS = [search_document, get_source_context, validate_question]
