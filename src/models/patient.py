"""
MedCollab — Patient Case Data Models

Represents a patient's clinical presentation including demographics,
symptoms, lab results, and medical history.
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class Symptom(BaseModel):
    """A single patient symptom with severity and duration."""

    name: str = Field(..., description="Symptom name, e.g. 'chest pain'")
    severity: Literal["mild", "moderate", "severe"] = Field(
        default="moderate", description="Symptom severity level"
    )
    duration: str = Field(
        default="unknown", description="Duration, e.g. '3 days', '2 weeks'"
    )
    location: Optional[str] = Field(
        default=None, description="Body location if applicable"
    )
    characteristics: list[str] = Field(
        default_factory=list,
        description="Descriptors, e.g. ['sharp', 'radiating', 'worsens with exertion']",
    )


class LabResult(BaseModel):
    """A single laboratory test result."""

    test_name: str = Field(..., description="Lab test name, e.g. 'Troponin-I'")
    value: float = Field(..., description="Numeric result value")
    unit: str = Field(..., description="Measurement unit, e.g. 'ng/mL'")
    reference_range: str = Field(
        default="", description="Normal range, e.g. '0.0–0.04 ng/mL'"
    )
    is_abnormal: bool = Field(
        default=False, description="Whether this value is outside normal range"
    )


class PatientCase(BaseModel):
    """
    Complete patient case — the primary input to the MedCollab pipeline.
    Can be constructed from MedQA questions or manually entered via UI.
    """

    patient_id: str = Field(default="P001", description="Unique patient identifier")
    age: int = Field(default=45, description="Patient age in years")
    sex: Literal["M", "F", "Other"] = Field(default="M", description="Patient sex")
    chief_complaint: str = Field(
        ..., description="Primary reason for visit in the patient's words"
    )
    symptoms: list[Symptom] = Field(
        default_factory=list, description="List of presenting symptoms"
    )
    lab_results: list[LabResult] = Field(
        default_factory=list, description="Available laboratory results"
    )
    medical_history: list[str] = Field(
        default_factory=list,
        description="Past medical conditions, e.g. ['Hypertension', 'Diabetes Type 2']",
    )
    current_medications: list[str] = Field(
        default_factory=list,
        description="Current medications, e.g. ['Metformin 500mg', 'Lisinopril 10mg']",
    )
    family_history: list[str] = Field(
        default_factory=list,
        description="Relevant family medical history",
    )
    social_history: str = Field(
        default="",
        description="Smoking, alcohol, occupation, etc.",
    )

    # ── Dynamic fields updated by Patient Interaction Agent ──
    additional_findings: list[str] = Field(
        default_factory=list,
        description="New evidence gathered via follow-up questions",
    )

    def summary(self) -> str:
        """Return a formatted text summary for LLM prompts."""
        parts = [
            f"Patient: {self.age}yo {self.sex}",
            f"Chief Complaint: {self.chief_complaint}",
        ]
        if self.symptoms:
            symptom_strs = [
                f"  - {s.name} ({s.severity}, {s.duration})"
                + (f" [{', '.join(s.characteristics)}]" if s.characteristics else "")
                for s in self.symptoms
            ]
            parts.append("Symptoms:\n" + "\n".join(symptom_strs))
        if self.lab_results:
            lab_strs = [
                f"  - {l.test_name}: {l.value} {l.unit}"
                + (" ⚠ ABNORMAL" if l.is_abnormal else "")
                for l in self.lab_results
            ]
            parts.append("Lab Results:\n" + "\n".join(lab_strs))
        if self.medical_history:
            parts.append(f"Medical History: {', '.join(self.medical_history)}")
        if self.current_medications:
            parts.append(f"Medications: {', '.join(self.current_medications)}")
        if self.family_history:
            parts.append(f"Family History: {', '.join(self.family_history)}")
        if self.social_history:
            parts.append(f"Social History: {self.social_history}")
        if self.additional_findings:
            parts.append(
                "Additional Findings (from follow-up):\n"
                + "\n".join(f"  - {f}" for f in self.additional_findings)
            )
        return "\n".join(parts)
