"""
MedCollab — Triage Agent

First agent in the pipeline. Analyzes the patient case, classifies
complexity, and dynamically recruits the most relevant specialists.
"""

from __future__ import annotations
import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from src.utils.llm import get_llm
from src.utils import extract_content
from src.config import AVAILABLE_SPECIALTIES, MIN_SPECIALISTS, MAX_SPECIALISTS
from src.prompts.triage import TRIAGE_SYSTEM_PROMPT, TRIAGE_USER_PROMPT

logger = logging.getLogger(__name__)


def triage_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: Triage Agent.

    Reads the patient case from state, classifies complexity,
    and selects specialists for consultation.

    Args:
        state: Current AgentState dict.

    Returns:
        Updated state with triage_result populated.
    """
    logger.info("🏥 Triage Agent: Analyzing patient case...")

    patient_case = state["patient_case"]
    patient_summary = patient_case.summary()

    llm = get_llm()
    messages = [
        SystemMessage(content=TRIAGE_SYSTEM_PROMPT.format(
            available_specialties=", ".join(AVAILABLE_SPECIALTIES)
        )),
        HumanMessage(content=TRIAGE_USER_PROMPT.format(
            patient_summary=patient_summary
        )),
    ]

    response = llm.invoke(messages)
    response_text = extract_content(response)

    # Parse JSON from response (handle markdown code blocks)
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()

    try:
        triage_result = json.loads(response_text)
    except json.JSONDecodeError:
        logger.error(f"Triage Agent: Failed to parse JSON. Raw response: {response_text}")
        # Fallback: recruit general_medicine + first available specialty
        triage_result = {
            "complexity": "moderate",
            "recruited_specialists": ["general_medicine", AVAILABLE_SPECIALTIES[0]],
            "reasoning": "Fallback triage due to parsing error",
            "key_symptoms": [],
            "primary_concern": patient_case.chief_complaint,
        }

    # Validate and clamp specialist count
    specialists = triage_result.get("recruited_specialists", [])
    specialists = [s for s in specialists if s in AVAILABLE_SPECIALTIES]

    if len(specialists) < MIN_SPECIALISTS:
        specialists = ["general_medicine", "cardiology"][:MIN_SPECIALISTS]
    elif len(specialists) > MAX_SPECIALISTS:
        specialists = specialists[:MAX_SPECIALISTS]

    triage_result["recruited_specialists"] = specialists

    logger.info(
        f"🏥 Triage: complexity={triage_result.get('complexity')}, "
        f"specialists={specialists}"
    )

    # Update state
    state_update = {
        "triage_result": triage_result,
        "messages": state.get("messages", []) + [
            {"role": "triage", "content": json.dumps(triage_result, indent=2)}
        ],
    }
    return state_update
