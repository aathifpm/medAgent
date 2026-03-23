"""
MedCollab — Evaluation Metrics

Measures diagnostic accuracy, reasoning consistency, and causal chain
validity for evaluation on MedQA and sample cases.
"""

from __future__ import annotations
import json
import logging
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)


def diagnostic_accuracy(predicted: str, ground_truth: str) -> bool:
    """
    Check if the predicted diagnosis matches the ground truth.
    Uses case-insensitive substring matching for flexibility.
    """
    pred = predicted.lower().strip()
    gt = ground_truth.lower().strip()
    return gt in pred or pred in gt


def consistency_score(specialist_positions: list[dict], final_diagnosis: str) -> float:
    """
    Measure how consistent specialist positions are with the final diagnosis.

    Returns: 0.0 (no agreement) to 1.0 (full agreement)
    """
    if not specialist_positions:
        return 0.0

    matching = 0
    for pos in specialist_positions:
        position = pos.get("position", "").lower()
        diagnosis = final_diagnosis.lower()
        if diagnosis in position or position in diagnosis:
            matching += 1

    return matching / len(specialist_positions)


def causal_chain_validity(causal_chain: dict) -> dict[str, Any]:
    """
    Evaluate the validity of the HDCC.

    Checks:
    - All links reference existing nodes
    - There is at least one disease-level node
    - Symptom nodes have evidence links
    - No orphan nodes (except root)
    """
    nodes = causal_chain.get("nodes", [])
    links = causal_chain.get("links", [])
    node_ids = {n["node_id"] for n in nodes}

    issues = []
    score = 1.0

    # Check link validity
    for link in links:
        if link["source_id"] not in node_ids:
            issues.append(f"Link source {link['source_id']} not in nodes")
            score -= 0.1
        if link["target_id"] not in node_ids:
            issues.append(f"Link target {link['target_id']} not in nodes")
            score -= 0.1

    # Check for disease nodes
    disease_nodes = [n for n in nodes if n.get("level") == "disease"]
    if not disease_nodes:
        issues.append("No disease-level nodes found")
        score -= 0.2

    # Check for orphan nodes
    linked_ids = set()
    for link in links:
        linked_ids.add(link["source_id"])
        linked_ids.add(link["target_id"])

    orphans = node_ids - linked_ids
    if orphans and len(nodes) > 1:
        issues.append(f"Orphan nodes: {orphans}")
        score -= 0.05 * len(orphans)

    # Check evidence links on symptom nodes
    symptom_nodes = [n for n in nodes if n.get("level") == "symptom"]
    no_evidence = [n for n in symptom_nodes if not n.get("evidence_links")]
    if no_evidence:
        issues.append(f"{len(no_evidence)} symptom nodes without evidence links")
        score -= 0.05 * len(no_evidence)

    return {
        "score": max(0.0, min(1.0, score)),
        "num_nodes": len(nodes),
        "num_links": len(links),
        "num_disease_nodes": len(disease_nodes),
        "num_orphans": len(orphans) if nodes else 0,
        "issues": issues,
        "valid": len(issues) == 0,
    }


def patient_interaction_value(
    diagnosis_with: str,
    diagnosis_without: str,
    ground_truth: str,
) -> dict[str, Any]:
    """
    Evaluate the value added by the Patient Interaction Agent.

    Compares diagnostic accuracy with and without follow-up questions.
    """
    acc_with = diagnostic_accuracy(diagnosis_with, ground_truth)
    acc_without = diagnostic_accuracy(diagnosis_without, ground_truth)

    return {
        "accuracy_with_interaction": acc_with,
        "accuracy_without_interaction": acc_without,
        "improvement": acc_with and not acc_without,
        "degradation": not acc_with and acc_without,
        "no_change": acc_with == acc_without,
    }


def batch_evaluate(results: list[dict]) -> dict[str, Any]:
    """
    Evaluate a batch of diagnostic pipeline results.

    Args:
        results: List of dicts with keys:
            - predicted: predicted diagnosis
            - ground_truth: ground truth diagnosis
            - specialist_positions: list of specialist position dicts
            - causal_chain: HDCC dict

    Returns:
        Aggregate metrics.
    """
    total = len(results)
    if total == 0:
        return {"error": "No results to evaluate"}

    correct = 0
    consistency_scores = []
    chain_scores = []

    for r in results:
        # Accuracy
        if diagnostic_accuracy(r["predicted"], r["ground_truth"]):
            correct += 1

        # Consistency
        cs = consistency_score(
            r.get("specialist_positions", []),
            r["predicted"],
        )
        consistency_scores.append(cs)

        # Chain validity
        cv = causal_chain_validity(r.get("causal_chain", {}))
        chain_scores.append(cv["score"])

    return {
        "total_cases": total,
        "accuracy": correct / total,
        "correct": correct,
        "avg_consistency": sum(consistency_scores) / len(consistency_scores),
        "avg_chain_validity": sum(chain_scores) / len(chain_scores),
    }
