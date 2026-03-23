"""
MedCollab — Causal Chain Builder Agent

Constructs the Hierarchical Disease Causal Chain (HDCC) from
specialist IBIS positions. Links symptoms → mechanisms → diseases.
"""

from __future__ import annotations
import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from src.utils.llm import get_llm
from src.utils import extract_content
from src.prompts.causal import CAUSAL_SYSTEM_PROMPT, CAUSAL_USER_PROMPT

logger = logging.getLogger(__name__)


def causal_builder_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: Causal Chain Builder.

    Takes all specialist IBIS positions and constructs an HDCC
    linking symptoms → mechanisms → diseases → comorbidities.

    Args:
        state: Current AgentState dict.

    Returns:
        Updated state with causal_chain populated.
    """
    logger.info("🔗 Causal Chain Builder: Constructing HDCC...")

    patient_case = state["patient_case"]
    patient_summary = patient_case.summary()
    specialist_positions = state.get("specialist_positions", [])

    # Format specialist positions for the prompt
    positions_text = json.dumps(specialist_positions, indent=2)

    llm = get_llm()
    messages = [
        SystemMessage(content=CAUSAL_SYSTEM_PROMPT),
        HumanMessage(content=CAUSAL_USER_PROMPT.format(
            patient_summary=patient_summary,
            specialist_positions=positions_text,
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
        causal_chain = json.loads(response_text)
    except json.JSONDecodeError:
        logger.error("Causal Chain Builder: Failed to parse JSON")
        causal_chain = {
            "nodes": [],
            "links": [],
            "root_diagnosis": "Unable to construct causal chain",
            "comorbidities": [],
            "summary": "Causal chain construction failed due to parsing error.",
        }

    logger.info(
        f"🔗 HDCC: root_diagnosis={causal_chain.get('root_diagnosis', 'N/A')}, "
        f"nodes={len(causal_chain.get('nodes', []))}, "
        f"links={len(causal_chain.get('links', []))}"
    )

    state_update = {
        "causal_chain": causal_chain,
        "messages": state.get("messages", []) + [
            {"role": "causal_builder", "content": json.dumps(causal_chain, indent=2)}
        ],
    }
    return state_update
