"""The Finding Assistant as an explicit LangGraph workflow.

Implements the flow as a ``StateGraph`` with six nodes and a conditional edge
after classification:

    classify_request
         │ (conditional edge)
     ┌───┴───────────────┐
  retrieval            direct
     │                    │
 retrieve_context         │
     └────────┬───────────┘
        generate_answer
              │
        extract_finding
              │
      validate_groundedness
              │
         return_report

Run directly for a demo:  ``python -m src.graph``
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .config import OUTPUTS_DIR
from .llm import get_chat_model
from .retriever import format_docs, get_retriever
from .schemas import Finding, RouteDecision, ValidationResult


class FindingAssistantState(TypedDict, total=False):
    """Workflow state shared across nodes."""

    question: str
    request_type: str
    documents: list
    context: str
    answer: str
    finding: dict
    validation_result: str
    final_report: str


# --- Nodes -----------------------------------------------------------------


def classify_request(state: FindingAssistantState) -> dict:
    """Decide whether the question needs document retrieval."""
    llm = get_chat_model().with_structured_output(RouteDecision)
    decision: RouteDecision = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Classify the user request for an IT Service Desk assistant.\n"
                    "Choose 'retrieval' for ANY question about an IT topic, problem, "
                    "or how-to (passwords, Wi-Fi, network, software, malware, email, "
                    "hardware, accounts, security, etc.) — these must be answered from "
                    "the BonBon FAQ.\n"
                    "Choose 'direct' ONLY for greetings, small talk, or meta questions "
                    "about the assistant itself (e.g. 'hello', 'what can you do?').\n"
                    "When in doubt, choose 'retrieval'."
                )
            ),
            HumanMessage(content=state["question"]),
        ]
    )
    return {"request_type": decision.request_type}


def retrieve_context(state: FindingAssistantState) -> dict:
    """Retrieve relevant chunks from the BonBon FAQ."""
    docs = get_retriever().invoke(state["question"])
    return {"documents": docs, "context": format_docs(docs)}


def generate_answer(state: FindingAssistantState) -> dict:
    """Generate an answer grounded in retrieved context."""
    context = state.get("context", "")
    llm = get_chat_model()
    if context:
        system = (
            "You are the BonBon IT Service Desk assistant. Answer using ONLY the "
            "context below. Cite the source page. If the context does not cover "
            "the question, say 'This is not covered in the BonBon FAQ.'\n\n"
            f"Context:\n{context}"
        )
    else:
        system = (
            "You are the BonBon IT Service Desk assistant. This is a direct "
            "request that needs no document lookup. Respond briefly."
        )
    answer = llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=state["question"])]
    )
    return {"answer": answer.content}


def extract_finding(state: FindingAssistantState) -> dict:
    """Extract a structured :class:`Finding` from the answer."""
    llm = get_chat_model().with_structured_output(Finding)
    finding: Finding = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Extract one structured finding. Base 'evidence' on the FAQ "
                    "context and pick severity (Low/Medium/High) by operational risk."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{state['question']}\n\n"
                    f"Answer:\n{state.get('answer', '')}\n\n"
                    f"Context:\n{state.get('context', '(none)')}"
                )
            ),
        ]
    )
    return {"finding": finding.model_dump()}


def validate_groundedness(state: FindingAssistantState) -> dict:
    """Quality gate: is the answer supported by the context?"""
    context = state.get("context", "")
    if not context:
        result = ValidationResult(
            is_grounded=False,
            reason="No document context was retrieved (direct request).",
            supporting_quote=None,
        )
    else:
        llm = get_chat_model().with_structured_output(ValidationResult)
        result = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a strict fact-checker. Decide whether the ANSWER is "
                        "fully supported by the CONTEXT. Set is_grounded accordingly "
                        "and quote supporting text when it exists."
                    )
                ),
                HumanMessage(
                    content=(
                        f"CONTEXT:\n{context}\n\nANSWER:\n{state.get('answer', '')}"
                    )
                ),
            ]
        )
    verdict = "GROUNDED" if result.is_grounded else "NOT GROUNDED"
    return {"validation_result": f"{verdict} — {result.reason}"}


def return_report(state: FindingAssistantState) -> dict:
    """Assemble the final human-readable report."""
    import json

    report = (
        f"QUESTION: {state['question']}\n"
        f"ROUTE: {state.get('request_type', 'unknown')}\n\n"
        f"ANSWER:\n{state.get('answer', '')}\n\n"
        f"STRUCTURED FINDING:\n{json.dumps(state.get('finding', {}), indent=2)}\n\n"
        f"GROUNDEDNESS: {state.get('validation_result', 'n/a')}"
    )
    return {"final_report": report}


# --- Routing ----------------------------------------------------------------


def route_after_classify(state: FindingAssistantState) -> str:
    """Conditional edge: retrieval questions fetch context, others skip it."""
    return "retrieve_context" if state.get("request_type") == "retrieval" else "generate_answer"


# --- Graph assembly ---------------------------------------------------------


def build_graph(with_memory: bool = True):
    """Build and compile the StateGraph workflow."""
    builder = StateGraph(FindingAssistantState)

    builder.add_node("classify_request", classify_request)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("extract_finding", extract_finding)
    builder.add_node("validate_groundedness", validate_groundedness)
    builder.add_node("return_report", return_report)

    builder.add_edge(START, "classify_request")
    builder.add_conditional_edges(
        "classify_request",
        route_after_classify,
        {"retrieve_context": "retrieve_context", "generate_answer": "generate_answer"},
    )
    builder.add_edge("retrieve_context", "generate_answer")
    builder.add_edge("generate_answer", "extract_finding")
    builder.add_edge("extract_finding", "validate_groundedness")
    builder.add_edge("validate_groundedness", "return_report")
    builder.add_edge("return_report", END)

    checkpointer = InMemorySaver() if with_memory else None
    return builder.compile(checkpointer=checkpointer)


def run(question: str, thread_id: str = "graph-demo") -> FindingAssistantState:
    """Run the workflow for a question and return the final state."""
    graph = build_graph()
    return graph.invoke(
        {"question": question},
        config={"configurable": {"thread_id": thread_id}},
    )


def export_graph_png(path: str | Path | None = None) -> Path:
    """Export the compiled graph as a PNG to ``outputs/graph.png``."""
    path = Path(path) if path else OUTPUTS_DIR / "graph.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = build_graph(with_memory=False)
    png_bytes = graph.get_graph().draw_mermaid_png()
    path.write_bytes(png_bytes)
    return path


def run_demo() -> None:
    """Demo: a retrieval question and a direct question (conditional edge)."""
    print("=" * 70)
    print("Finding Assistant — Assignment 2 (LangGraph) demo")
    print("=" * 70)

    for q, tid in [
        ("My computer is infected with malware. What should I do?", "g1"),
        ("Hello, what can you help me with?", "g2"),
    ]:
        print(f"\n### Question: {q}")
        state = run(q, thread_id=tid)
        print(state["final_report"])
        print("-" * 70)

    try:
        out = export_graph_png()
        print(f"\nGraph image written to: {out}")
    except Exception as exc:  # draw_mermaid_png needs network/graphviz backend
        print(f"\n(Graph PNG export skipped: {exc})")


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    run_demo()
