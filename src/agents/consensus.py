"""
MedCollab — Consensus / GP Agent

GP-led consensus mechanism with weighted voting and logic auditing.
Iteratively filters low-quality reasoning by penalizing inconsistent agents.
"""

from __future__ import annotations
import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from src.utils.llm import get_llm
from src.utils import extract_content
from src.config import CONSENSUS_CONFIDENCE_THRESHOLD, MAX_CONSENSUS_ROUNDS
from src.prompts.consensus import CONSENSUS_SYSTEM_PROMPT, CONSENSUS_USER_PROMPT

logger = logging.getLogger(__name__)


def consensus_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: Consensus / GP Agent.

    Reviews all specialist positions and causal chain,
    scores each agent's reasoning, performs weighted voting,
    and determines if consensus is reached.

    Args:
        state: Current AgentState dict.

    Returns:
        Updated state with consensus_result and is_consensus_reached.
    """
    current_round = state.get("current_round", 1)
    max_rounds = state.get("max_rounds", MAX_CONSENSUS_ROUNDS)

    logger.info(f"⚖️ Consensus Agent: Round {current_round}/{max_rounds}...")

    patient_case = state["patient_case"]
    patient_summary = patient_case.summary()
    specialist_positions = state.get("specialist_positions", [])
    causal_chain = state.get("causal_chain", {})
    follow_up_questions = state.get("follow_up_questions", [])
    follow_up_answers = state.get("follow_up_answers", [])

    # Format follow-up context
    follow_up_context = ""
    if follow_up_questions:
        qa_pairs = []
        for i, (q, a) in enumerate(zip(follow_up_questions, follow_up_answers)):
            qa_pairs.append(f"  Q{i+1}: {q}\n  A{i+1}: {a}")
        follow_up_context = (
            "Follow-up questions asked by the Patient Interaction Agent:\n"
            + "\n".join(qa_pairs)
        )
    else:
        follow_up_context = "No follow-up questions were asked."

    # Format previous consensus attempts
    previous_consensus = state.get("previous_consensus_attempts", [])
    prev_text = "None" if not previous_consensus else json.dumps(previous_consensus, indent=2)

    causal_summary = causal_chain.get("summary", json.dumps(causal_chain, indent=2))
    positions_text = json.dumps(specialist_positions, indent=2)

    llm = get_llm()
    messages = [
        SystemMessage(content=CONSENSUS_SYSTEM_PROMPT.format(
            consensus_threshold=CONSENSUS_CONFIDENCE_THRESHOLD,
        )),
        HumanMessage(content=CONSENSUS_USER_PROMPT.format(
            patient_summary=patient_summary,
            specialist_positions=positions_text,
            causal_chain_summary=causal_summary,
            follow_up_context=follow_up_context,
            round_number=current_round,
            max_rounds=max_rounds,
            previous_consensus=prev_text,
        )),
    ]

    response = llm.invoke(messages)
    response_text = extract_content(response)

    # Parse JSON
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()

    try:
        consensus_result = json.loads(response_text)
    except json.JSONDecodeError:
        logger.error("Consensus Agent: Failed to parse JSON")
        consensus_result = {
            "consensus_reached": True,  # force end to prevent infinite loop
            "consensus_score": 0.5,
            "primary_diagnosis": "Unable to reach consensus — review needed",
            "differential_diagnoses": [],
            "agent_scores": [],
            "vote_distribution": {},
            "recommendations": ["Manual clinical review recommended"],
            "dissenting_agents": [],
            "reasoning": "Consensus parsing failed, forcing termination.",
        }

    is_consensus = consensus_result.get("consensus_reached", False)
    consensus_score = consensus_result.get("consensus_score", 0.0)

    # Force consensus on final round
    if current_round >= max_rounds and not is_consensus:
        logger.warning(f"⚖️ Forcing consensus on final round {current_round}")
        is_consensus = True
        consensus_result["consensus_reached"] = True

    logger.info(
        f"⚖️ Consensus: reached={is_consensus}, "
        f"score={consensus_score:.2f}, "
        f"diagnosis={consensus_result.get('primary_diagnosis', 'N/A')}"
    )

    # Track previous attempts for next round (if needed)
    prev_attempts = list(previous_consensus)
    prev_attempts.append({
        "round": current_round,
        "score": consensus_score,
        "diagnosis": consensus_result.get("primary_diagnosis", ""),
    })

    state_update = {
        "consensus_result": consensus_result,
        "is_consensus_reached": is_consensus,
        "current_round": current_round + 1,
        "previous_consensus_attempts": prev_attempts,
        "messages": state.get("messages", []) + [
            {"role": "consensus", "content": json.dumps(consensus_result, indent=2)}
        ],
    }
    return state_update
