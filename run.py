"""
MedCollab — CLI Entry Point

Run the diagnostic pipeline from the command line with sample patient cases.

Usage:
    python run.py                          # Run case 0
    python run.py --case-index 2           # Run case 2
    python run.py --case-file my_case.json # Run custom case
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.patient import PatientCase
from src.graph.workflow import run_diagnosis

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)


def load_sample_case(index: int = 0) -> tuple[PatientCase, str]:
    """Load a sample case from data/sample_cases.json."""
    data_path = Path(__file__).parent / "data" / "sample_cases.json"
    with open(data_path, "r") as f:
        cases = json.load(f)

    if index >= len(cases):
        console.print(f"[red]Error: Case index {index} out of range (0-{len(cases)-1})[/red]")
        sys.exit(1)

    case_data = cases[index]
    ground_truth = case_data.pop("ground_truth", "")
    patient_case = PatientCase(**case_data)
    return patient_case, ground_truth


def load_custom_case(filepath: str) -> tuple[PatientCase, str]:
    """Load a custom case from a JSON file."""
    with open(filepath, "r") as f:
        case_data = json.load(f)

    ground_truth = case_data.pop("ground_truth", "")
    patient_case = PatientCase(**case_data)
    return patient_case, ground_truth


def display_results(final_state: dict) -> None:
    """Pretty-print the diagnostic results using Rich."""
    consensus = final_state.get("consensus_result", {})
    causal_chain = final_state.get("causal_chain", {})
    specialist_positions = final_state.get("specialist_positions", [])
    follow_up_q = final_state.get("follow_up_questions", [])
    follow_up_a = final_state.get("follow_up_answers", [])

    # ── Header ──
    console.print()
    console.print(Panel.fit(
        f"[bold green]🏥 MedCollab Diagnostic Report[/bold green]",
        border_style="green",
    ))

    # ── Triage ──
    triage = final_state.get("triage_result", {})
    console.print(Panel(
        f"[bold]Complexity:[/bold] {triage.get('complexity', 'N/A')}\n"
        f"[bold]Specialists:[/bold] {', '.join(triage.get('recruited_specialists', []))}\n"
        f"[bold]Primary Concern:[/bold] {triage.get('primary_concern', 'N/A')}",
        title="🏥 Triage Result",
        border_style="cyan",
    ))

    # ── Specialist Positions ──
    table = Table(title="🩺 Specialist IBIS Positions", border_style="blue")
    table.add_column("Specialist", style="cyan")
    table.add_column("Position", style="green")
    table.add_column("Confidence", style="yellow", justify="center")
    table.add_column("Differentials", style="dim")

    for pos in specialist_positions:
        table.add_row(
            pos.get("agent_name", "Unknown"),
            pos.get("position", "N/A"),
            f"{pos.get('confidence', 0):.0%}",
            ", ".join(pos.get("differential_diagnoses", [])[:3]),
        )
    console.print(table)

    # ── Patient Interaction (NOVEL) ──
    if follow_up_q:
        tree = Tree("🤔 [bold]Patient Interaction Agent (Novel)[/bold]")
        for q, a in zip(follow_up_q, follow_up_a):
            branch = tree.add(f"[cyan]Q: {q}[/cyan]")
            branch.add(f"[green]A: {a}[/green]")
        console.print(tree)

    # ── Causal Chain ──
    console.print(Panel(
        f"[bold]Root Diagnosis:[/bold] {causal_chain.get('root_diagnosis', 'N/A')}\n"
        f"[bold]Nodes:[/bold] {len(causal_chain.get('nodes', []))}\n"
        f"[bold]Links:[/bold] {len(causal_chain.get('links', []))}\n"
        f"[bold]Comorbidities:[/bold] {', '.join(causal_chain.get('comorbidities', ['None']))}\n"
        f"[bold]Summary:[/bold] {causal_chain.get('summary', 'N/A')}",
        title="🔗 Hierarchical Disease Causal Chain",
        border_style="yellow",
    ))

    # ── Final Diagnosis ──
    diag = consensus.get("primary_diagnosis", "N/A")
    score = consensus.get("consensus_score", 0)
    recs = consensus.get("recommendations", [])

    console.print(Panel(
        f"[bold white on green] {diag} [/bold white on green]\n\n"
        f"[bold]Consensus Score:[/bold] {score:.0%}\n"
        f"[bold]Differentials:[/bold] {', '.join(consensus.get('differential_diagnoses', []))}\n"
        f"[bold]Recommendations:[/bold]\n" + "\n".join(f"  • {r}" for r in recs),
        title="✅ Final Diagnosis",
        border_style="green",
    ))

    # ── Ground Truth Comparison ──
    ground_truth = final_state.get("ground_truth", "")
    if ground_truth:
        match = ground_truth.lower() in diag.lower() or diag.lower() in ground_truth.lower()
        status = "[green]✅ MATCH[/green]" if match else "[red]❌ MISMATCH[/red]"
        console.print(Panel(
            f"[bold]Ground Truth:[/bold] {ground_truth}\n"
            f"[bold]Predicted:[/bold] {diag}\n"
            f"[bold]Status:[/bold] {status}",
            title="📊 Evaluation",
            border_style="magenta",
        ))


def main():
    parser = argparse.ArgumentParser(description="MedCollab — Multi-Agent Medical Diagnosis")
    parser.add_argument("--case-index", type=int, default=0, help="Sample case index (0-4)")
    parser.add_argument("--case-file", type=str, default=None, help="Custom case JSON file")
    parser.add_argument("--max-rounds", type=int, default=3, help="Max consensus rounds")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print(Panel.fit(
        "[bold cyan]🏥 MedCollab — Multi-Agent Medical Diagnosis System[/bold cyan]\n"
        "[dim]Causal-Driven Multi-Agent Collaboration via IBIS Argumentation[/dim]",
        border_style="cyan",
    ))

    # Load patient case
    if args.case_file:
        patient_case, ground_truth = load_custom_case(args.case_file)
        console.print(f"[cyan]Loading custom case from: {args.case_file}[/cyan]")
    else:
        patient_case, ground_truth = load_sample_case(args.case_index)
        console.print(f"[cyan]Loading sample case #{args.case_index}[/cyan]")

    console.print(Panel(patient_case.summary(), title="📋 Patient Case", border_style="white"))

    # Run pipeline
    final_state = run_diagnosis(
        patient_case=patient_case,
        ground_truth=ground_truth,
        max_rounds=args.max_rounds,
    )

    # Display results
    display_results(final_state)


if __name__ == "__main__":
    main()
