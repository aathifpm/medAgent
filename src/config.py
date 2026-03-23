"""
MedCollab — Configuration Module

Loads settings from .env and exposes them as module-level constants.
Swap LLM provider by changing LLM_PROVIDER in your .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# ── LLM Provider ─────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")

# OpenAI
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

# Ollama (local)
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")

# Anthropic
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

# Google
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL: str = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")

# Azure OpenAI
AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")

# ── General Settings ─────────────────────────────────────────
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
MAX_CONSENSUS_ROUNDS: int = int(os.getenv("MAX_CONSENSUS_ROUNDS", "3"))

# ── Specialist Roster ────────────────────────────────────────
AVAILABLE_SPECIALTIES: list[str] = [
    "cardiology",
    "neurology",
    "pulmonology",
    "gastroenterology",
    "endocrinology",
    "nephrology",
    "infectious_disease",
    "rheumatology",
    "hematology",
    "general_medicine",
]

# ── Consensus Thresholds ─────────────────────────────────────
CONSENSUS_CONFIDENCE_THRESHOLD: float = 0.7
MIN_SPECIALISTS: int = 2
MAX_SPECIALISTS: int = 4
