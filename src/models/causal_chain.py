"""
MedCollab — Hierarchical Disease Causal Chain (HDCC) Models

Models pathological progression through explicit causal and comorbidity
relationships: Symptom → Mechanism → Disease → Comorbidity.
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class CausalChainNode(BaseModel):
    """A single node in the Hierarchical Disease Causal Chain."""

    node_id: str = Field(..., description="Unique node identifier, e.g. 'N01'")
    label: str = Field(
        ..., description="Human-readable label, e.g. 'Elevated Troponin-I'"
    )
    level: Literal["symptom", "mechanism", "disease", "comorbidity"] = Field(
        ..., description="Hierarchy level in the causal chain"
    )
    description: str = Field(
        default="", description="Detailed explanation of this node"
    )
    causes: list[str] = Field(
        default_factory=list,
        description="IDs of parent nodes that cause/lead to this node",
    )
    evidence_links: list[str] = Field(
        default_factory=list,
        description="Patient evidence supporting this node (symptom names, lab values)",
    )


class CausalLink(BaseModel):
    """An explicit causal relationship between two nodes."""

    source_id: str = Field(..., description="ID of the cause node")
    target_id: str = Field(..., description="ID of the effect node")
    relationship: Literal["causes", "contributes_to", "comorbid_with", "indicates"] = Field(
        ..., description="Type of causal relationship"
    )
    strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Strength of the causal relationship",
    )


class HDCC(BaseModel):
    """
    Hierarchical Disease Causal Chain — the structured representation
    of pathological progression constructed by the Causal Chain Builder.

    Structure:
        Symptoms (leaf) → Mechanisms → Disease (root) → Comorbidities (linked)
    """

    nodes: list[CausalChainNode] = Field(
        default_factory=list, description="All nodes in the causal chain"
    )
    links: list[CausalLink] = Field(
        default_factory=list, description="Causal relationships between nodes"
    )
    root_diagnosis: str = Field(
        default="", description="Primary disease node label"
    )
    comorbidities: list[str] = Field(
        default_factory=list, description="Identified comorbid conditions"
    )
    summary: str = Field(
        default="",
        description="Natural language summary of the causal chain",
    )

    def get_node(self, node_id: str) -> Optional[CausalChainNode]:
        """Look up a node by ID."""
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def symptom_nodes(self) -> list[CausalChainNode]:
        """Return all symptom-level nodes."""
        return [n for n in self.nodes if n.level == "symptom"]

    def disease_nodes(self) -> list[CausalChainNode]:
        """Return all disease-level nodes."""
        return [n for n in self.nodes if n.level == "disease"]
