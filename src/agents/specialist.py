"""
MedCollab — Specialist Agent Factory

Creates specialist agents dynamically based on the specialty type.
Each specialist follows IBIS protocol and references previous positions
for richer cross-agent debate.
"""

from __future__ import annotations
import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from src.utils.llm import get_llm
from src.utils import extract_content
from src.models.ibis import IBISPosition, IBISArgument
from src.prompts.specialist import (
    SPECIALIST_SYSTEM_PROMPT,
    SPECIALIST_USER_PROMPT,
    PREVIOUS_POSITIONS_TEMPLATE,
    SPECIALTY_FOCUS,
    SPECIALTY_TITLES,
)

logger = logging.getLogger(__name__)


def _format_previous_positions(positions: list[dict]) -> str:
    """Format existing specialist positions for context injection."""
    if not positions:
        return "No other specialists have provided positions yet. You are the first."

    formatted = []
    for pos in positions:
        formatted.append(
            f"  [{pos.get('agent_name', 'Unknown')}]\n"
            f"  Position: {pos.get('position', 'N/A')}\n"
            f"  Confidence: {pos.get('confidence', 'N/A')}\n"
            f"  Key Evidence: {', '.join(pos.get('evidence_links', pos.get('arguments_for', [{}])[0].get('evidence', ['N/A']) if pos.get('arguments_for') else ['N/A']))}"
        )

    return PREVIOUS_POSITIONS_TEMPLATE.format(
        positions="\n\n".join(formatted)
    )


def _parse_ibis_position(response_text: str, specialty: str) -> dict:
    """Parse LLM response into IBISPosition dict with error handling."""
    # Strip markdown code blocks
    text = response_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"Specialist {specialty}: Failed to parse JSON")
        data = {
            "agent_name": SPECIALTY_TITLES.get(specialty, specialty),
            "specialty": specialty,
            "issue": "Unable to parse structured response",
            "position": "Further evaluation needed",
            "confidence": 0.3,
            "arguments_for": [],
            "arguments_against": [],
            "differential_diagnoses": [],
            "recommended_tests": [],
        }

    return data


def specialist_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: Specialist Agents.

    Iterates over recruited specialists and collects IBIS positions.
    Each specialist can see previous positions for richer debate.

    Args:
        state: Current AgentState dict.

    Returns:
        Updated state with specialist_positions populated.
    """
    triage_result = state.get("triage_result", {})
    recruited = triage_result.get("recruited_specialists", ["general_medicine"])
    patient_case = state["patient_case"]
    patient_summary = patient_case.summary()

    existing_positions = state.get("specialist_positions", [])
    all_positions = list(existing_positions)  # carry forward from previous rounds

    llm = get_llm()

    for specialty in recruited:
        title = SPECIALTY_TITLES.get(specialty, specialty.replace("_", " ").title())
        focus = SPECIALTY_FOCUS.get(specialty, "the overall clinical picture")

        logger.info(f"🩺 {title}: Analyzing patient case...")

        prev_context = _format_previous_positions(all_positions)

        messages = [
            SystemMessage(content=SPECIALIST_SYSTEM_PROMPT.format(
                specialty_title=title,
                specialty=specialty,
                specialty_focus=focus,
            )),
            HumanMessage(content=SPECIALIST_USER_PROMPT.format(
                specialty_title=title,
                patient_summary=patient_summary,
                previous_positions_context=prev_context,
            )),
        ]

        response = llm.invoke(messages)
        position_data = _parse_ibis_position(extract_content(response), specialty)

        # Ensure required fields
        position_data.setdefault("agent_name", title)
        position_data.setdefault("specialty", specialty)

        all_positions.append(position_data)
        logger.info(
            f"🩺 {title}: Position = {position_data.get('position', 'N/A')} "
            f"(confidence: {position_data.get('confidence', 'N/A')})"
        )

    state_update = {
        "specialist_positions": all_positions,
        "messages": state.get("messages", []) + [
            {"role": "specialists", "content": json.dumps(all_positions, indent=2)}
        ],
    }
    return state_update
