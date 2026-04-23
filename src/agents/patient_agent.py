"""
MedCollab — Patient Interaction Agent (NOVEL CONTRIBUTION)

This agent adds a dynamic follow-up querying loop to the diagnostic process.
It analyzes evidence gaps in specialist positions and generates targeted
follow-up questions. In automated mode, it simulates patient responses.

This is the primary research contribution — extending MedCollab from
static input to iterative, interactive diagnosis.
"""

from __future__ import annotations
import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from src.utils.llm import get_llm
from src.utils import extract_content
from src.prompts.patient import (
    PATIENT_AGENT_SYSTEM_PROMPT,
    PATIENT_AGENT_USER_PROMPT,
    SIMULATED_PATIENT_SYSTEM_PROMPT,
    SIMULATED_PATIENT_USER_PROMPT,
)

logger = logging.getLogger(__name__)


def _analyze_evidence_gaps(
    patient_summary: str,
    specialist_positions: list[dict],
    causal_chain: dict,
) -> dict:
    """Call LLM to identify evidence gaps and generate follow-up questions."""
    llm = get_llm()

    causal_summary = causal_chain.get("summary", "Not yet constructed")
    positions_text = json.dumps(specialist_positions, indent=2)

    messages = [
        SystemMessage(content=PATIENT_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=PATIENT_AGENT_USER_PROMPT.format(
            patient_summary=patient_summary,
            specialist_positions=positions_text,
            causal_chain_summary=causal_summary,
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
        result = json.loads(response_text)
    except json.JSONDecodeError:
        logger.error("Patient Agent: Failed to parse evidence gap analysis")
        result = {
            "has_evidence_gaps": False,
            "evidence_gaps": [],
            "follow_up_questions": [],
            "reasoning": "Unable to analyze evidence gaps due to parsing error",
        }

    return result


def _simulate_patient_responses(
    patient_case: Any,
    questions: list[str],
    ground_truth: str = "",
) -> dict:
    """Simulate patient responses for automated evaluation."""
    llm = get_llm()

    patient_summary = patient_case.summary()
    questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

    messages = [
        SystemMessage(content=SIMULATED_PATIENT_SYSTEM_PROMPT.format(
            ground_truth_diagnosis=ground_truth or "Unknown — respond based on symptoms",
        )),
        HumanMessage(content=SIMULATED_PATIENT_USER_PROMPT.format(
            age=patient_case.age,
            sex=patient_case.sex,
            patient_summary=patient_summary,
            questions=questions_text,
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
        result = json.loads(response_text)
    except json.JSONDecodeError:
        logger.error("Simulated Patient: Failed to parse response")
        result = {
            "answers": [
                {"question": q, "answer": "I'm not sure.", "reveals_new_evidence": False, "new_evidence": ""}
                for q in questions
            ]
        }

    return result


def patient_interaction_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: Patient Interaction Agent (NOVEL).

    Analyzes evidence gaps, generates follow-up questions,
    simulates patient responses, and updates the patient case
    with new evidence.

    Args:
        state: Current AgentState dict.

    Returns:
        Updated state with follow-up Q&A and potentially updated patient case.
    """
    logger.info("🤔 Patient Interaction Agent: Analyzing evidence gaps...")

    patient_case = state["patient_case"]
    patient_summary = patient_case.summary()
    specialist_positions = state.get("specialist_positions", [])
    causal_chain = state.get("causal_chain", {})
    ground_truth = state.get("ground_truth", "")

    # Step 1: Analyze evidence gaps
    gap_analysis = _analyze_evidence_gaps(
        patient_summary, specialist_positions, causal_chain
    )

    has_gaps = gap_analysis.get("has_evidence_gaps", False)
    questions = gap_analysis.get("follow_up_questions", [])

    follow_up_questions = state.get("follow_up_questions", [])
    follow_up_answers = state.get("follow_up_answers", [])
    new_evidence_items = []

    if has_gaps and questions:
        logger.info(f"🤔 Patient Agent: Found {len(questions)} evidence gaps. Asking follow-ups...")

        # Step 2: Get patient responses (simulated in automated mode)
        patient_responses = _simulate_patient_responses(
            patient_case, questions, ground_truth
        )

        answers = patient_responses.get("answers", [])
        for ans in answers:
            q = ans.get("question", "")
            a = ans.get("answer", "")
            follow_up_questions.append(q)
            follow_up_answers.append(a)

            # Collect new evidence
            if ans.get("reveals_new_evidence") and ans.get("new_evidence"):
                new_evidence_items.append(ans["new_evidence"])
                logger.info(f"🆕 New evidence: {ans['new_evidence']}")

        # Step 3: Update patient case with new evidence
        if new_evidence_items:
            updated_findings = list(patient_case.additional_findings) + new_evidence_items
            patient_case = patient_case.model_copy(
                update={"additional_findings": updated_findings}
            )
    else:
        logger.info("🤔 Patient Agent: No significant evidence gaps found.")

    patient_interaction_round = state.get("patient_interaction_round", 1)

    state_update = {
        "patient_case": patient_case,
        "follow_up_questions": follow_up_questions,
        "follow_up_answers": follow_up_answers,
        "has_new_evidence": len(new_evidence_items) > 0,
        "patient_interaction_round": patient_interaction_round + 1,
        "messages": state.get("messages", []) + [
            {"role": "patient_agent", "content": json.dumps({
                "has_gaps": has_gaps,
                "questions_asked": len(questions),
                "new_evidence_found": len(new_evidence_items),
                "evidence_items": new_evidence_items,
                "patient_interaction_round": patient_interaction_round,
            }, indent=2)}
        ],
    }
    return state_update
