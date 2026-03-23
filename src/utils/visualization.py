"""
MedCollab — Causal Chain Visualization

Renders the HDCC as a networkx graph for both CLI (matplotlib)
and Streamlit display.
"""

from __future__ import annotations
import io
from typing import Optional

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Color scheme for node hierarchy levels
LEVEL_COLORS = {
    "symptom": "#FF6B6B",       # Red
    "mechanism": "#FFA726",     # Orange
    "disease": "#66BB6A",       # Green
    "comorbidity": "#42A5F5",   # Blue
}

LEVEL_SHAPES = {
    "symptom": "o",       # Circle
    "mechanism": "s",     # Square
    "disease": "D",       # Diamond
    "comorbidity": "^",   # Triangle
}


def build_nx_graph(causal_chain: dict) -> nx.DiGraph:
    """Convert HDCC dict to a networkx DiGraph."""
    G = nx.DiGraph()

    nodes = causal_chain.get("nodes", [])
    links = causal_chain.get("links", [])

    for node in nodes:
        G.add_node(
            node["node_id"],
            label=node.get("label", node["node_id"]),
            level=node.get("level", "symptom"),
            description=node.get("description", ""),
        )

    for link in links:
        G.add_edge(
            link["source_id"],
            link["target_id"],
            relationship=link.get("relationship", "causes"),
            strength=link.get("strength", 0.5),
        )

    return G


def render_causal_chain(
    causal_chain: dict,
    title: str = "Hierarchical Disease Causal Chain (HDCC)",
    figsize: tuple = (14, 8),
    save_path: Optional[str] = None,
) -> Optional[bytes]:
    """
    Render the HDCC as a matplotlib figure.

    Args:
        causal_chain: HDCC dict.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save to this path.

    Returns:
        PNG bytes if save_path is None, else None.
    """
    G = build_nx_graph(causal_chain)

    if len(G.nodes) == 0:
        return None

    fig, ax = plt.subplots(figsize=figsize, facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Layout
    try:
        pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
    except Exception:
        pos = nx.shell_layout(G)

    # Draw edges
    edge_colors = []
    for u, v, data in G.edges(data=True):
        strength = data.get("strength", 0.5)
        edge_colors.append(strength)

    if G.edges():
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color="#555555",
            width=1.5,
            alpha=0.7,
            arrows=True,
            arrowsize=20,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
        )

    # Draw nodes by level
    for level, color in LEVEL_COLORS.items():
        node_list = [n for n, d in G.nodes(data=True) if d.get("level") == level]
        if node_list:
            nx.draw_networkx_nodes(
                G, pos, ax=ax,
                nodelist=node_list,
                node_color=color,
                node_size=800,
                alpha=0.9,
                edgecolors="white",
                linewidths=1.5,
            )

    # Draw labels
    labels = {n: d.get("label", n) for n, d in G.nodes(data=True)}
    # Wrap long labels
    wrapped_labels = {}
    for k, v in labels.items():
        if len(v) > 20:
            words = v.split()
            mid = len(words) // 2
            wrapped_labels[k] = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        else:
            wrapped_labels[k] = v

    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels=wrapped_labels,
        font_size=8,
        font_color="white",
        font_weight="bold",
    )

    # Legend
    patches = [
        mpatches.Patch(color=color, label=level.capitalize())
        for level, color in LEVEL_COLORS.items()
    ]
    ax.legend(
        handles=patches,
        loc="upper left",
        fontsize=10,
        facecolor="#16213e",
        edgecolor="white",
        labelcolor="white",
    )

    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=20)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
        return None
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
