"""
MedCollab — IBIS Argumentation Models

Implements the Issue-Based Information System (IBIS) protocol.
Every agent's diagnostic position must be backed by traceable evidence
and structured as: Issue → Position → Arguments (for/against) → Evidence.
"""

from __future__ import annotations
from pydantic import BaseModel, Field


class IBISArgument(BaseModel):
    """A single argument supporting or opposing a diagnostic position."""

    claim: str = Field(..., description="The argument statement")
    evidence: list[str] = Field(
        default_factory=list,
        description="Traceable evidence supporting this argument (symptoms, lab values, guidelines)",
    )
    strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Argument strength: 0.0 (very weak) to 1.0 (very strong)",
    )


class IBISPosition(BaseModel):
    """
    An agent's diagnostic position following IBIS protocol.

    Structure:
        Issue: "What is causing the patient's chest pain?"
        Position: "Acute Coronary Syndrome"
        Arguments For: [evidence-backed claims supporting this]
        Arguments Against: [evidence-backed claims against this]
    """

    agent_name: str = Field(
        ..., description="Name of the agent, e.g. 'Cardiologist'"
    )
    specialty: str = Field(
        ..., description="Agent's specialty area, e.g. 'cardiology'"
    )
    issue: str = Field(
        ...,
        description="The diagnostic question being addressed, "
        "e.g. 'What is causing the chest pain?'",
    )
    position: str = Field(
        ...,
        description="The agent's diagnostic stance, e.g. 'Acute Coronary Syndrome'",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall confidence in this position: 0.0–1.0",
    )
    arguments_for: list[IBISArgument] = Field(
        default_factory=list,
        description="Arguments supporting this diagnostic position",
    )
    arguments_against: list[IBISArgument] = Field(
        default_factory=list,
        description="Arguments against this diagnostic position (self-critique)",
    )
    differential_diagnoses: list[str] = Field(
        default_factory=list,
        description="Alternative diagnoses considered and why they were less likely",
    )
    recommended_tests: list[str] = Field(
        default_factory=list,
        description="Additional tests the specialist recommends to confirm/rule out",
    )

    def logic_score(self) -> float:
        """
        Compute a logic consistency score.
        Higher = more evidence for the position relative to evidence against.
        Used by the Consensus Agent for weighted voting.
        """
        total_for = sum(a.strength for a in self.arguments_for) if self.arguments_for else 0
        total_against = sum(a.strength for a in self.arguments_against) if self.arguments_against else 0
        total = total_for + total_against
        if total == 0:
            return 0.5
        return total_for / total
