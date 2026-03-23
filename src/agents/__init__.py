"""MedCollab — Agents Package"""

from src.agents.triage import triage_agent
from src.agents.specialist import specialist_agent
from src.agents.causal_builder import causal_builder_agent
from src.agents.patient_agent import patient_interaction_agent
from src.agents.consensus import consensus_agent

__all__ = [
    "triage_agent",
    "specialist_agent",
    "causal_builder_agent",
    "patient_interaction_agent",
    "consensus_agent",
]
