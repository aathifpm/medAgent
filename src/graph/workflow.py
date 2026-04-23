"""
MedCollab — LangGraph Workflow Definition

Builds the full diagnostic pipeline as a LangGraph StateGraph:

    START → Triage → Specialists → Causal Chain → Patient Interaction
         ↕ (loop if new evidence)       ↕ (loop if no consensus)
    Patient Interaction → Consensus → END

Key conditional edges:
  1. Patient Agent → Specialists: if new evidence found, loop back
  2. Patient Agent → Consensus: if no gaps, proceed to voting
  3. Consensus → Specialists: if consensus < threshold, retry
  4. Consensus → END: if consensus reached or max rounds hit
"""

from __future__ import annotations
import logging
from typing import Any, Literal

from langgraph.graph import StateGraph, END

from src.graph.state import AgentState
from src.agents.triage import triage_agent
from src.agents.specialist import specialist_agent
from src.agents.causal_builder import causal_builder_agent
from src.agents.patient_agent import patient_interaction_agent
from src.agents.consensus import consensus_agent
from src.config import MAX_CONSENSUS_ROUNDS

logger = logging.getLogger(__name__)


# ── Conditional Edge Functions ───────────────────────────────

def should_loop_patient(state: AgentState) -> Literal["specialists", "consensus"]:
    """
    After Patient Interaction Agent:
    - If new evidence was found → loop back to specialists for re-evaluation
    - If no gaps → proceed to consensus
    """
    has_new_evidence = state.get("has_new_evidence", False)
    patient_interaction_round = state.get("patient_interaction_round", 1)

    if has_new_evidence and patient_interaction_round <= 1:
        # Only loop back once to avoid infinite evidence gathering
        logger.info("↩️ New evidence found — looping back to specialists")
        return "specialists"
    else:
        logger.info("➡️ Proceeding to consensus")
        return "consensus"


def should_loop_consensus(state: AgentState) -> Literal["specialists", "end"]:
    """
    After Consensus Agent:
    - If consensus not reached and rounds remain → loop back to specialists
    - If consensus reached or max rounds → end
    """
    is_consensus = state.get("is_consensus_reached", False)
    current_round = state.get("current_round", 1)
    max_rounds = state.get("max_rounds", MAX_CONSENSUS_ROUNDS)

    if is_consensus:
        logger.info("✅ Consensus reached — finishing")
        return "end"
    elif current_round > max_rounds:
        logger.info("⏰ Max rounds reached — forcing finish")
        return "end"
    else:
        logger.info(f"🔄 No consensus — round {current_round}/{max_rounds}, retrying")
        return "specialists"


# ── Graph Builder ────────────────────────────────────────────

def build_workflow() -> StateGraph:
    """
    Build and compile the MedCollab StateGraph.

    Returns:
        Compiled LangGraph ready for invocation.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("triage", triage_agent)
    workflow.add_node("specialists", specialist_agent)
    workflow.add_node("causal_builder", causal_builder_agent)
    workflow.add_node("patient_interaction", patient_interaction_agent)
    workflow.add_node("consensus", consensus_agent)

    # Set entry point
    workflow.set_entry_point("triage")

    # Linear edges
    workflow.add_edge("triage", "specialists")
    workflow.add_edge("specialists", "causal_builder")
    workflow.add_edge("causal_builder", "patient_interaction")

    # Conditional edge: Patient Interaction → Specialists OR Consensus
    workflow.add_conditional_edges(
        "patient_interaction",
        should_loop_patient,
        {
            "specialists": "specialists",
            "consensus": "consensus",
        },
    )

    # Conditional edge: Consensus → Specialists (retry) OR END
    workflow.add_conditional_edges(
        "consensus",
        should_loop_consensus,
        {
            "specialists": "specialists",
            "end": END,
        },
    )

    # Compile
    app = workflow.compile()
    logger.info("📊 MedCollab workflow compiled successfully")

    return app


def run_diagnosis(
    patient_case: Any,
    ground_truth: str = "",
    max_rounds: int = MAX_CONSENSUS_ROUNDS,
) -> dict:
    """
    Run the full MedCollab diagnostic pipeline.

    Args:
        patient_case: A PatientCase instance.
        ground_truth: Ground truth diagnosis (for evaluation/simulation).
        max_rounds: Maximum consensus rounds.

    Returns:
        Final AgentState dict with all results.
    """
    app = build_workflow()

    initial_state: AgentState = {
        "patient_case": patient_case,
        "ground_truth": ground_truth,
        "triage_result": {},
        "specialist_positions": [],
        "causal_chain": {},
        "follow_up_questions": [],
        "follow_up_answers": [],
        "has_new_evidence": False,
        "patient_interaction_round": 1,
        "consensus_result": {},
        "current_round": 1,
        "max_rounds": max_rounds,
        "is_consensus_reached": False,
        "previous_consensus_attempts": [],
        "messages": [],
    }

    logger.info("🚀 Starting MedCollab diagnostic pipeline...")
    final_state = app.invoke(initial_state)
    logger.info("🏁 MedCollab pipeline complete!")

    return final_state
