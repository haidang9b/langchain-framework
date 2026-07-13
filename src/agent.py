"""The Finding Assistant as a RAG agent.

Uses LangChain v1's ``create_agent`` wired with:

* retrieval tools over the BonBon FAQ,
* an ``InMemorySaver`` checkpointer for thread-based short-term memory,
* a separate structured-output call to extract a :class:`Finding`.

Run directly for a demo:  ``python -m src.agent``
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from .llm import get_chat_model
from .retriever import format_docs, get_retriever
from .schemas import Finding
from .tools import ALL_TOOLS

SYSTEM_PROMPT = (
    "You are the BonBon IT Service Desk Finding Assistant.\n"
    "Answer questions ONLY using information found in the BonBon FAQ document.\n"
    "Always call the search_document tool to retrieve context before answering.\n"
    "Ground every answer in the retrieved context and cite the source page like "
    "(source: BonBon FAQ.pdf, page: N).\n"
    "If the retrieved context does not cover the question, reply exactly: "
    "'This is not covered in the BonBon FAQ.'\n"
    "Be concise and practical, as an IT support agent would be."
)

# A module-level agent so memory (the checkpointer) persists across calls
# within a process. Built lazily to keep imports cheap and env-free.
_AGENT = None


def get_agent():
    """Build (once) and return the compiled ReAct agent with memory."""
    global _AGENT
    if _AGENT is None:
        _AGENT = create_agent(
            model=get_chat_model(),
            tools=ALL_TOOLS,
            checkpointer=InMemorySaver(),
            system_prompt=SYSTEM_PROMPT,
        )
    return _AGENT


def answer_question(question: str, thread_id: str = "default") -> str:
    """Answer a question, remembering prior turns on the same ``thread_id``.

    The checkpointer keys conversation state by ``thread_id``: reuse an id to
    continue a conversation, use a fresh id to start a clean one.
    """
    agent = get_agent()
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content


def extract_finding(question: str, answer: str, context: str | None = None) -> Finding:
    """Extract a structured :class:`Finding` from a Q/A pair.

    A dedicated ``with_structured_output`` call guarantees schema-valid JSON.
    If ``context`` is not supplied, it is retrieved fresh for the question.
    """
    if context is None:
        context = format_docs(get_retriever().invoke(question))

    structured_llm = get_chat_model().with_structured_output(Finding)
    messages = [
        SystemMessage(
            content=(
                "Extract a single structured finding from the IT support answer "
                "below. Base 'evidence' strictly on the provided FAQ context and "
                "choose severity from Low/Medium/High according to operational risk."
            )
        ),
        HumanMessage(
            content=(
                f"Question:\n{question}\n\n"
                f"Answer:\n{answer}\n\n"
                f"FAQ context:\n{context}"
            )
        ),
    ]
    return structured_llm.invoke(messages)


def run_demo() -> None:
    """Demo: grounded Q&A, a structured finding, and thread-based memory."""

    print("=" * 70)
    print("Finding Assistant — Assignment 1 demo")
    print("=" * 70)

    # --- Grounded question + structured finding -----------------------------
    q1 = "How do I reset my password?"
    print(f"\n[Q] {q1}")
    a1 = answer_question(q1, thread_id="demo-conv")
    print(f"[A] {a1}")

    finding = extract_finding(q1, a1)
    print("\n[Structured Finding]")
    print(finding.model_dump_json(indent=2))

    # --- Short-term memory: same thread remembers ---------------------------
    print("\n" + "-" * 70)
    print("Memory demo — SAME thread_id ('demo-conv') should recall context:")
    follow_up = "Can you summarize what we just discussed?"
    print(f"[Q] {follow_up}")
    print(f"[A] {answer_question(follow_up, thread_id='demo-conv')}")

    # --- Different thread has no memory -------------------------------------
    print("\n" + "-" * 70)
    print("Memory demo — DIFFERENT thread_id ('fresh') should NOT recall it:")
    print(f"[Q] {follow_up}")
    print(f"[A] {answer_question(follow_up, thread_id='fresh')}")


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    run_demo()
