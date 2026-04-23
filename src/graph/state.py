"""
MedCollab — Agent State Definition

Shared state that flows through the LangGraph StateGraph.
All agents read from and write to this state.
"""

from __future__ import annotations
from typing import Any, TypedDict
from src.models.patient import PatientCase


class AgentState(TypedDict, total=False):
    """
    Shared state for the MedCollab LangGraph pipeline.

    Fields are populated progressively as each agent executes.
    Using total=False so agents only need to return their updated fields.
    """

    # ── Input ────────────────────────────────────────────────
    patient_case: PatientCase           # The patient case being diagnosed
    ground_truth: str                   # Ground truth diagnosis (for evaluation)

    # ── Triage ───────────────────────────────────────────────
    triage_result: dict                 # Complexity + recruited specialists

    # ── Specialists ──────────────────────────────────────────
    specialist_positions: list[dict]    # List of IBIS position dicts

    # ── Causal Chain ─────────────────────────────────────────
    causal_chain: dict                  # HDCC dict

    # ── Patient Interaction (NOVEL) ──────────────────────────
    follow_up_questions: list[str]      # Questions asked by Patient Agent
    follow_up_answers: list[str]        # Answers from patient/simulation
    has_new_evidence: bool              # Whether follow-up yielded new info

    # ── Consensus ────────────────────────────────────────────
    consensus_result: dict              # Final consensus dict
    current_round: int                  # Current consensus round (1-indexed)
    max_rounds: int                     # Maximum consensus rounds
    is_consensus_reached: bool          # Whether threshold was met
    previous_consensus_attempts: list[dict]  # History of consensus rounds

    # ── Patient interaction loop guard ───────────────────────
    patient_interaction_round: int       # Patient interaction loop iteration (1-indexed)

    # ── Trace ────────────────────────────────────────────────
    messages: list[dict]                # Full conversation trace for debugging
