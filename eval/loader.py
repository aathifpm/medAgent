"""
MedCollab — MedQA Dataset Loader

Converts MedQA (USMLE-style) questions into PatientCase format
for evaluation with the MedCollab pipeline.

MedQA Dataset: https://github.com/jind11/MedQA
Download: The dataset should be placed in data/medqa/
"""

from __future__ import annotations
import json
import re
import logging
from pathlib import Path
from typing import Any

from src.models.patient import PatientCase, Symptom

logger = logging.getLogger(__name__)


def parse_medqa_question(question: dict) -> tuple[PatientCase, str]:
    """
    Convert a single MedQA question into a PatientCase.

    MedQA format:
    {
        "question": "A 45-year-old man presents with...",
        "answer": "B",
        "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
        "meta_info": "..."
    }

    Returns:
        (PatientCase, ground_truth_answer)
    """
    q_text = question.get("question", "")
    answer_key = question.get("answer", "")
    options = question.get("options", {})
    ground_truth = options.get(answer_key, answer_key)

    # Extract demographics from question text
    age = _extract_age(q_text)
    sex = _extract_sex(q_text)

    # Build patient case
    patient_case = PatientCase(
        patient_id=f"MedQA_{hash(q_text) % 10000:04d}",
        age=age,
        sex=sex,
        chief_complaint=q_text[:200],  # First 200 chars as chief complaint
        symptoms=_extract_symptoms(q_text),
        medical_history=_extract_history(q_text),
        social_history=_extract_social(q_text),
    )

    return patient_case, ground_truth


def _extract_age(text: str) -> int:
    """Extract age from question text."""
    patterns = [
        r"(\d+)-year-old",
        r"(\d+)\s*year\s*old",
        r"age\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 45  # default


def _extract_sex(text: str) -> str:
    """Extract sex from question text."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["woman", "female", " she ", "her "]):
        return "F"
    elif any(w in text_lower for w in ["man", "male", " he ", "his "]):
        return "M"
    return "M"  # default


def _extract_symptoms(text: str) -> list[Symptom]:
    """Extract basic symptoms from question text."""
    # Common symptom keywords
    symptom_keywords = [
        "pain", "fever", "cough", "headache", "nausea", "vomiting",
        "fatigue", "dyspnea", "shortness of breath", "dizziness",
        "weakness", "swelling", "rash", "diarrhea", "constipation",
        "chest pain", "abdominal pain", "back pain", "joint pain",
    ]

    symptoms = []
    text_lower = text.lower()
    for keyword in symptom_keywords:
        if keyword in text_lower:
            symptoms.append(Symptom(
                name=keyword,
                severity="moderate",
                duration="unknown",
            ))

    return symptoms


def _extract_history(text: str) -> list[str]:
    """Extract medical history from question text."""
    history_keywords = [
        "diabetes", "hypertension", "asthma", "COPD", "cancer",
        "heart disease", "stroke", "kidney disease", "liver disease",
    ]
    text_lower = text.lower()
    return [h for h in history_keywords if h in text_lower]


def _extract_social(text: str) -> str:
    """Extract social history from question text."""
    social_items = []
    text_lower = text.lower()
    if "smok" in text_lower:
        social_items.append("Smoker")
    if "alcohol" in text_lower or "drink" in text_lower:
        social_items.append("Alcohol use")
    if "drug" in text_lower:
        social_items.append("Drug use history")
    return ", ".join(social_items)


def load_medqa_dataset(
    data_dir: str = "data/medqa",
    split: str = "test",
    max_cases: int = 100,
) -> list[tuple[PatientCase, str]]:
    """
    Load MedQA dataset and convert to PatientCase format.

    Args:
        data_dir: Path to MedQA dataset directory.
        split: Dataset split (train/dev/test).
        max_cases: Maximum number of cases to load.

    Returns:
        List of (PatientCase, ground_truth) tuples.
    """
    data_path = Path(data_dir) / f"{split}.jsonl"

    if not data_path.exists():
        # Try alternative paths
        alt_paths = [
            Path(data_dir) / f"questions/{split}.jsonl",
            Path(data_dir) / f"{split}.json",
        ]
        for alt in alt_paths:
            if alt.exists():
                data_path = alt
                break
        else:
            logger.warning(f"MedQA dataset not found at {data_path}. Using sample cases instead.")
            return []

    cases = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_cases:
                break
            try:
                question = json.loads(line.strip())
                case, gt = parse_medqa_question(question)
                cases.append((case, gt))
            except Exception as e:
                logger.warning(f"Skipping MedQA question {i}: {e}")

    logger.info(f"Loaded {len(cases)} MedQA cases from {data_path}")
    return cases
