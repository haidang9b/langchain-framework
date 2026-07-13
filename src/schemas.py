"""Pydantic schemas for structured output.

These models are passed to ``llm.with_structured_output(...)`` so the model
is constrained to emit valid, typed JSON — no manual parsing.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Severity = Literal["Low", "Medium", "High"]


class Finding(BaseModel):
    """A structured finding extracted from an answer.

    Severity is constrained to the priority levels used in the BonBon FAQ
    (Low / Medium / High).
    """

    title: str = Field(description="Short, specific title of the finding.")
    severity: Severity = Field(description="Priority level: Low, Medium, or High.")
    evidence: str = Field(
        description="The factual basis for the finding, grounded in the FAQ context."
    )
    recommendation: str = Field(
        description="Concrete recommended action to resolve or address the finding."
    )


class ValidationResult(BaseModel):
    """Result of the groundedness quality gate."""

    is_grounded: bool = Field(
        description="True if the answer is fully supported by the retrieved context."
    )
    reason: str = Field(description="Brief explanation of the groundedness verdict.")
    supporting_quote: Optional[str] = Field(
        default=None,
        description="A short quote from the context that supports the answer, if any.",
    )


class RouteDecision(BaseModel):
    """Routing decision for the graph's classify node."""

    request_type: Literal["retrieval", "direct"] = Field(
        description=(
            "'retrieval' if answering needs the BonBon FAQ document; "
            "'direct' for greetings/meta questions that need no document lookup."
        )
    )
    reason: str = Field(description="Why this route was chosen.")
