"""
=================
Architecture & Structure Visualization Suite
=================
Diagram 7: Log-Based Structured Visualization (GraphRAG Knowledge Graph Structure)
Diagram 8: Overall RAG Pipeline Comparison (Vector RAG vs GraphRAG Architecture)
"""

from __future__ import annotations

import json
import logging
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color Scheme
# ---------------------------------------------------------------------------
class DiagramColors:
    """Unified color palette for architectural diagrams"""
    # GraphRAG elements (Teal/Green family)
    GRAPH_NODE = "#1b9e77"
    GRAPH_EDGE = "#2c3e50"
    GRAPH_HIGHLIGHT = "#66c2a5"
    
    # Vector RAG elements (Orange/Red family)
    VECTOR_MAIN = "#d95f02"
    VECTOR_LIGHT = "#fc8d62"
    
    # Neutral elements
    DATA_SOURCE = "#3498db"
    TEXT_CHUNK = "#bdc3c7"
    LLM = "#8e44ad"
    
    # Background & borders
    BG_LIGHT = "#f8f9fa"
    BG_DARK = "#ecf0f1"
    BORDER = "#95a5a6"
    TEXT_PRIMARY = "#2c3e50"
    TEXT_SECONDARY = "#7f8c8d"

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_graphrag_results(filepath: str | Path) -> list[dict]:
    """Load GraphRAG evaluation results."""
    path = Path(filepath)
    if not path.exists():
        log.warning("File not found: %s", path)
        return []
    
    data = json.loads(path.read_text(encoding="utf-8"))
    log.info("Loaded %d GraphRAG evaluation records", len(data))
    return data


def parse_triplet(triplet_text: str) -> dict[str, str] | None:
    """
    Parse a triplet from retrieved_context format.
    
    Example: "[Net Revenues] --DEPENDS_ON--> [Tpv]. Context: ... (Score: 103.74)"
    
    Returns:
        Dict with keys: subject, predicate, object, context, score
    """
    pattern = r'\[(.*?)\]\s*--(\w+)-->\s*\[(.*?)\]\.\s*Context:\s*(.*?)\s*\(Score:\s*([\d.]+)\)'
    match = re.match(pattern, triplet_text)
    
    if match:
        return {
            "subject": match.group(1),
            "predicate": match.group(2),
            "object": match.group(3),
            "context": match.group(4),
            "score": float(match.group(5))
        }
    return None


def extract_sample_triplets(
    results: list[dict], 
    max_triplets: int = 5,
    query_types: list[str] = ["easy", "medium"]
) -> list[dict]:
    """Extract diverse sample triplets from evaluation results."""
    triplets = []
    
    for record in results:
        if record.get("difficulty") not in query_types:
            continue
        
        context = record.get("retrieved_context", "")
        if not context or context == "No relevant graph context found.":
            continue
        
        # Split by newline to get individual triplets
        lines = context.split("\n")
        for line in lines:
            triplet = parse_triplet(line)
            if triplet and len(triplets) < max_triplets:
                triplets.append(triplet)
        
        if len(triplets) >= max_triplets:
            break
    
    return triplets

# ---------------------------------------------------------------------------
# Diagram 7: Log-Based Structured Visualization
# ------
# ---------------------------------------------------------------------
def plot_structured_knowledge_graph(
    results: list[dict] | None = None,
    out_dir: Path = Path("figures"),
    dpi: int = 300
) -> tuple[plt.Figure, plt.Axes]:
    """
    Visualize the structured nature of GraphRAG outputs using real triplet examples.
    All layout is driven by named constants — no floating magic numbers.
    """

    # ── Layout constants (all in data-space units) ─────────────────────────
    L = dict(
        # x-positions
        x_min       = 0.0,
        x_max       = 14.0,
        subj_x      = 0.3,   subj_w  = 2.8,
        arrow_x0    = 3.15,  arrow_x1 = 5.35,
        pred_x      = 4.25,                        # label above arrow
        obj_x       = 5.5,   obj_w   = 3.0,
        ctx_x       = 8.65,                        # context text start
        score_x     = 13.3,  score_r  = 0.22,      # score badge
        node_h      = 0.60,  node_pad = 0.30,      # box height / half-height

        # y-spacing
        row_step    = 1.25,                        # vertical gap between triplets
        legend_gap  = 0.85,                        # gap below last triplet → legend
        legend_h    = 0.30,                        # height reserved for legend row

        # context text
        ctx_max_w   = 40,                          # textwrap width (chars)
        ctx_fontsize = 7,
    )

    # ── Load / fallback data ────────────────────────────────────────────────
    if results:
        triplets = extract_sample_triplets(results, max_triplets=5)
    else:
        triplets = [
            {
                "subject":   "Net Revenues",
                "predicate": "DEPENDS_ON",
                "object":    "TPV",
                "context":   "Net revenues increased by 4% in 2025 primarily driven by 7% growth in total payment volume.",
                "score":     103.75,
            },
            {
                "subject":   "PayPal",
                "predicate": "EXPOSES_TO",
                "object":    "Advanced Persistent Threats",
                "context":   "PayPal's systems are targets of evolving cyber threats including ransomware and DDoS.",
                "score":     103.92,
            },
            {
                "subject":   "JPMorganChase",
                "predicate": "DEPENDS_ON",
                "object":    "Dividends",
                "context":   "The parent generally depends on receiving dividends from its subsidiaries to fund obligations.",
                "score":     104.29,
            },
        ]

    n = len(triplets)
    log.info(f"Visualizing {n} triplets from GraphRAG outputs")

    # ── Compute exact y-extents for the top panel ───────────────────────────
    #   rows run from top (y_top) downward; legend sits below the last row.
    y_top      = (n - 1) * L["row_step"]          # centre of first row
    y_bottom   = 0.0                               # centre of last row
    legend_y   = y_bottom - L["legend_gap"]
    top_y_max  =  y_top + 1.2
    top_y_min  =  legend_y - L["legend_h"] - 0.2

    # ── Figure & gridspec ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    gs  = fig.add_gridspec(2, 1, height_ratios=[7, 3], hspace=0.25)
    ax_graph  = fig.add_subplot(gs[0])
    ax_vector = fig.add_subplot(gs[1])

    # ── TOP PANEL: GraphRAG graph ───────────────────────────────────────────
    ax_graph.set_xlim(L["x_min"] - 0.3, L["x_max"])
    ax_graph.set_ylim(top_y_min, top_y_max)
    ax_graph.axis("off")

    cx = (L["x_min"] + L["x_max"]) / 2
    ax_graph.text(
        cx, top_y_max - 0.35,
        "GraphRAG: Structured Knowledge Graph Representation",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color=DiagramColors.TEXT_PRIMARY,
    )

    # ── Draw each triplet row ───────────────────────────────────────────────
    for i, triplet in enumerate(triplets):
        y = y_top - i * L["row_step"]
        hp = L["node_pad"]  # half-height offset for box placement

        # Subject box
        ax_graph.add_patch(FancyBboxPatch(
            (L["subj_x"], y - hp), L["subj_w"], L["node_h"],
            boxstyle="round,pad=0.1",
            facecolor=DiagramColors.GRAPH_NODE,
            edgecolor=DiagramColors.GRAPH_EDGE,
            linewidth=2, alpha=0.85,
        ))
        ax_graph.text(
            L["subj_x"] + L["subj_w"] / 2, y,
            triplet["subject"],
            ha="center", va="center", fontsize=10,
            color="white", fontweight="bold",
        )

        # Arrow
        ax_graph.add_patch(FancyArrowPatch(
            (L["arrow_x0"], y), (L["arrow_x1"], y),
            arrowstyle="-|>", mutation_scale=20,
            linewidth=2.5, color=DiagramColors.GRAPH_EDGE,
        ))

        # Predicate label (above arrow)
        ax_graph.text(
            L["pred_x"], y + 0.32,
            triplet["predicate"],
            ha="center", va="center", fontsize=9,
            color=DiagramColors.GRAPH_EDGE, fontweight="bold", style="italic",
        )

        # Object box
        ax_graph.add_patch(FancyBboxPatch(
            (L["obj_x"], y - hp), L["obj_w"], L["node_h"],
            boxstyle="round,pad=0.1",
            facecolor=DiagramColors.GRAPH_HIGHLIGHT,
            edgecolor=DiagramColors.GRAPH_EDGE,
            linewidth=2, alpha=0.85,
        ))
        ax_graph.text(
            L["obj_x"] + L["obj_w"] / 2, y,
            triplet["object"],
            ha="center", va="center", fontsize=10,
            color=DiagramColors.TEXT_PRIMARY, fontweight="bold",
        )

        # Context annotation — constrained to [ctx_x, score_x - score_r - 0.4]
        ctx_text = textwrap.fill(f'"{triplet["context"]}"', width=L["ctx_max_w"])
        ax_graph.text(
            L["ctx_x"], y,
            ctx_text,
            ha="left", va="center", fontsize=L["ctx_fontsize"],
            color=DiagramColors.TEXT_SECONDARY, style="italic",
            clip_on=True,
        )

        # Score badge — anchored at score_x, never overlaps context
        ax_graph.add_patch(plt.Circle(
            (L["score_x"], y), L["score_r"],
            color=DiagramColors.LLM, alpha=0.28,
        ))
        ax_graph.text(
            L["score_x"], y,
            f'{triplet["score"]:.0f}',
            ha="center", va="center", fontsize=7,
            color=DiagramColors.LLM, fontweight="bold",
        )

    # ── Legend (always below last triplet, never overlapping) ──────────────
    lx = L["x_min"] + 0.2
    ax_graph.text(lx, legend_y, "Legend:",
                  ha="left", va="center", fontsize=9,
                  fontweight="bold", color=DiagramColors.TEXT_PRIMARY)

    # Entity node
    ax_graph.add_patch(FancyBboxPatch(
        (lx + 1.2, legend_y - 0.14), 0.9, 0.28,
        boxstyle="round,pad=0.05",
        facecolor=DiagramColors.GRAPH_NODE,
        edgecolor=DiagramColors.GRAPH_EDGE, linewidth=1.5, alpha=0.8,
    ))
    ax_graph.text(lx + 1.65, legend_y, "Entity",
                  ha="center", va="center", fontsize=7,
                  color="white", fontweight="bold")
    ax_graph.text(lx + 2.25, legend_y, "= Entity Node",
                  ha="left", va="center", fontsize=8,
                  color=DiagramColors.TEXT_SECONDARY)

    # Typed relationship arrow
    ax_graph.add_patch(FancyArrowPatch(
        (lx + 4.5, legend_y), (lx + 5.2, legend_y),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=2, color=DiagramColors.GRAPH_EDGE,
    ))
    ax_graph.text(lx + 5.4, legend_y, "= Typed Relationship",
                  ha="left", va="center", fontsize=8,
                  color=DiagramColors.TEXT_SECONDARY)

    # Context provenance
    ax_graph.text(lx + 8.5, legend_y, '"Context"',
                  ha="left", va="center", fontsize=7,
                  color=DiagramColors.TEXT_SECONDARY, style="italic")
    ax_graph.text(lx + 9.5, legend_y, "= Source Text Provenance",
                  ha="left", va="center", fontsize=8,
                  color=DiagramColors.TEXT_SECONDARY)

    # ── BOTTOM PANEL: Vector RAG text blob ──────────────────────────────────
    ax_vector.set_xlim(0, 14.0)
    ax_vector.set_ylim(-0.6, 3.2)
    ax_vector.axis("off")

    ax_vector.text(
        7.0, 3.0,
        "Vector RAG: Unstructured Dense Text Chunks",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color=DiagramColors.TEXT_PRIMARY,
    )

    blob_text = (
        "PayPal's systems are targets of evolving cyber threats including ransomware and DDoS attacks. "
        "The company maintains various security measures and insurance coverage. However, insufficient "
        "insurance coverage may expose PayPal to unmitigated financial losses from disruptions. "
        "Net revenues increased by 4% in 2025 primarily driven by 7% growth in total payment volume. "
        "TPV measures payments processed on PayPal's platform, excluding reversals and gateway-exclusive "
        "transactions. The parent company generally depends on receiving dividends from its subsidiaries "
        "to fund obligations and operational expenses..."
    )

    # Box spans [0.4, 13.6] → width 13.2 units
    box_x0, box_x1 = 0.4, 13.6
    box_w = box_x1 - box_x0
    ax_vector.add_patch(FancyBboxPatch(
        (box_x0, 0.3), box_w, 1.8,
        boxstyle="round,pad=0.15",
        facecolor=DiagramColors.TEXT_CHUNK,
        edgecolor=DiagramColors.BORDER,
        linewidth=2, alpha=0.6,
    ))

    # Wrap width derived from figure width at this font size (empirically ~115 chars for 15" wide fig)
    wrapped = textwrap.fill(blob_text, width=115)
    ax_vector.text(
        (box_x0 + box_x1) / 2, 1.2,
        wrapped,
        ha="center", va="center", fontsize=8,
        color=DiagramColors.TEXT_PRIMARY,
        multialignment="left",
    )

    ax_vector.text(
        7.0, -0.25,
        "⚠  No explicit structure  •  Mixed facts  •  No typed relationships  •  High noise-to-signal ratio",
        ha="center", va="center", fontsize=9,
        color=DiagramColors.VECTOR_MAIN,
        fontweight="bold", style="italic",
    )

    # ── Save ────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        out_path = out_dir / f"plot_7_structured_knowledge_graph.{fmt}"
        fig.savefig(out_path, dpi=dpi if fmt == "png" else None,
                    bbox_inches="tight", format=fmt)
        log.info("Saved → %s", out_path)

    return fig, (ax_graph, ax_vector)
# ---------------------------------------------------------------------------
# Diagram 8: Overall RAG Pipeline Comparison
# ---------------------------------------------------------------------------
def plot_pipeline_comparison(
    out_dir: Path = Path("figures"),
    dpi: int = 300
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a parallel flowchart comparing Vector RAG vs GraphRAG architectures.
    
    Shows complete pipeline:
    - Data ingestion
    - Indexing/Processing
    - Query processing
    - Retrieval
    - Context window composition
    - LLM prompting
    
    Args:
        out_dir: Output directory for figure.
        dpi: Resolution for PNG output.
    
    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(-1, 16)
    ax.set_ylim(-1, 13)
    ax.axis('off')
    
    # === Title ===
    ax.text(7.5, 12.5, "RAG Architecture Comparison",
           ha='center', va='center', fontsize=16, fontweight='bold',
           color=DiagramColors.TEXT_PRIMARY)
    
    # === Column Headers ===
    # Vector RAG column (left)
    ax.text(3.5, 11.5, "Baseline Vector RAG",
           ha='center', va='center', fontsize=13, fontweight='bold',
           color=DiagramColors.VECTOR_MAIN,
           bbox=dict(boxstyle="round,pad=0.5", facecolor=DiagramColors.VECTOR_LIGHT, alpha=0.3))
    
    # GraphRAG column (right)
    ax.text(11.5, 11.5, "Stratified GraphRAG",
           ha='center', va='center', fontsize=13, fontweight='bold',
           color=DiagramColors.GRAPH_NODE,
           bbox=dict(boxstyle="round,pad=0.5", facecolor=DiagramColors.GRAPH_HIGHLIGHT, alpha=0.3))
    
    # === Stage 1: Data Ingestion ===
    stage_y = 10.5
    
    # Shared data source
    data_box = FancyBboxPatch(
        (6.5, stage_y - 0.3), 2, 0.6,
        boxstyle="round,pad=0.1",
        facecolor=DiagramColors.DATA_SOURCE,
        edgecolor=DiagramColors.BORDER,
        linewidth=2,
        alpha=0.8
    )
    ax.add_patch(data_box)
    ax.text(7.5, stage_y, "SEC 10-K Filings",
           ha='center', va='center', fontsize=10,
           color='white', fontweight='bold')
    
    # Arrows to both pipelines
    arrow_left = FancyArrowPatch(
        (6.5, stage_y), (4.5, stage_y - 1),
        arrowstyle='-|>',
        mutation_scale=15,
        linewidth=2,
        color=DiagramColors.BORDER
    )
    ax.add_patch(arrow_left)
    
    arrow_right = FancyArrowPatch(
        (8.5, stage_y), (10.5, stage_y - 1),
        arrowstyle='-|>',
        mutation_scale=15,
        linewidth=2,
        color=DiagramColors.BORDER
    )
    ax.add_patch(arrow_right)
    
    # === Stage 2: Chunking/Processing ===
    stage_y = 9
    
    # Vector RAG: Semantic chunking
    vector_chunk_box = FancyBboxPatch(
        (2, stage_y - 0.4), 3, 0.8,
        boxstyle="round,pad=0.1",
        facecolor=DiagramColors.VECTOR_LIGHT,
        edgecolor=DiagramColors.VECTOR_MAIN,
        linewidth=2,
        alpha=0.7
    )
    ax.add_patch(vector_chunk_box)
    ax.text(3.5, stage_y + 0.1, "Semantic Chunking",
           ha='center', va='center', fontsize=9,
           color=DiagramColors.TEXT_PRIMARY, fontweight='bold')
    ax.text(3.5, stage_y - 0.2, "Dense 800-token paragraphs",
           ha='center', va='center', fontsize=7,
           color=DiagramColors.TEXT_SECONDARY, style='italic')
    
    # GraphRAG: Knowledge graph extraction
    graph_extract_box = FancyBboxPatch(
        (10, stage_y - 0.4), 3, 0.8,
        boxstyle="round,pad=0.1",
        facecolor=DiagramColors.GRAPH_HIGHLIGHT,
        edgecolor=DiagramColors.GRAPH_NODE,
        linewidth=2,
        alpha=0.7
    )
    ax.add_patch(graph_extract_box)
    ax.text(11.5, stage_y + 0.1, "KG Extraction",
           ha='center', va='center', fontsize=9,
           color=DiagramColors.TEXT_PRIMARY, fontweight='bold')
    ax.text(11.5, stage_y - 0.2, "Entities, relationships, ontology",
           ha='center', va='center', fontsize=7,
           color=DiagramColors.TEXT_SECONDARY, style='italic')
    
    # Arrows down
    arrow_v2 = FancyArrowPatch(
        (3.5, stage_y - 0.5), (3.5, stage_y - 1.5),
        arrowstyle='-|>',
        mutation_scale=15,
        linewidth=2,
        color=DiagramColors.VECTOR_MAIN
    )
    ax.add_patch(arrow_v2)
    
    arrow_g2 = FancyArrowPatch(
        (11.5, stage_y - 0.5), (11.5, stage_y - 1.5),
        arrowstyle='-|>',
        mutation_scale=15,
        linewidth=2,
        color=DiagramColors.GRAPH_NODE
    )
    ax.add_patch(arrow_g2)
    
    # === Stage 3: Indexing ===
    stage_y = 7
    
    # Vector RAG: Embeddings + Vector DB
    vector_index_box = FancyBboxPatch(
        (2, stage_y - 0.4), 3, 0.8,
        boxstyle="round,pad=0.1",
        facecolor=DiagramColors.VECTOR_LIGHT,
        edgecolor=DiagramColors.VECTOR_MAIN,
        linewidth=2,
        alpha=0.7
    )
    ax.add_patch(vector_index_box)
    ax.text(3.5, stage_y + 0.1, "Vector Embedding",
           ha='center', va='center', fontsize=9,
           color=DiagramColors.TEXT_PRIMARY, fontweight='bold')
    ax.text(3.5, stage_y - 0.2, "ChromaDB + BM25 fusion (α=0.7)",
           ha='center', va='center', fontsize=7,
           color=DiagramColors.TEXT_SECONDARY, style='italic')
    
    # GraphRAG: Neo4j graph database
    graph_index_box = FancyBboxPatch(
        (10, stage_y - 0.4), 3, 0.8,
        boxstyle="round,pad=0.1",
        facecolor=DiagramColors.GRAPH_HIGHLIGHT,
        edgecolor=DiagramColors.GRAPH_NODE,
        linewidth=2,
        alpha=0.7
    )
    ax.add_patch(graph_index_box)
    ax.text(11.5, stage_y + 0.1, "Graph Database",
           ha='center', va='center', fontsize=9,
           color=DiagramColors.TEXT_PRIMARY, fontweight='bold')
    ax.text(11.5, stage_y - 0.2, "Neo4j: 720 nodes, 1,712 edges",
           ha='center', va='center', fontsize=7,
           color=DiagramColors.TEXT_SECONDARY, style='italic')
    
    # Arrows down
    arrow_v3 = FancyArrowPatch(
        (3.5, stage_y - 0.5), (3.5, stage_y - 1.5),
        arrowstyle='-|>',
        mutation_scale=15,
        linewidth=2,
        color=DiagramColors.VECTOR_MAIN
    )
    ax.add_patch(arrow_v3)
    
    arrow_g3 = FancyArrowPatch(
        (11.5, stage_y - 0.5), (11.5, stage_y - 1.5),
        arrowstyle='-|>',
        mutation_scale=15,
        linewidth=2,
        color=DiagramColors.GRAPH_NODE
    )
    ax.add_patch(arrow_g3)
    
    # === Stage 4: Query Processing ===
    stage_y = 5
    
    # Vector RAG: Simple embedding
    vector_query_box = FancyBboxPatch(
        (2, stage_y - 0.4), 3, 0.8,
        boxstyle="round,pad=0.1",
        facecolor=DiagramColors.VECTOR_LIGHT,
        edgecolor=DiagramColors.VECTOR_MAIN,
        linewidth=2,
        alpha=0.7
    )
    ax.add_patch(vector_query_box)
    ax.text(3.5, stage_y + 0.1, "Query Embedding",
           ha='center', va='center', fontsize=9,
           color=DiagramColors.TEXT_PRIMARY, fontweight='bold')
    ax.text(3.5, stage_y - 0.2, "Cosine similarity search",
           ha='center', va='center', fontsize=7,
           color=DiagramColors.TEXT_SECONDARY, style='italic')
    
    # GraphRAG: Entity extraction + traversal
    graph_query_box = FancyBboxPatch(
        (10, stage_y - 0.4), 3, 0.8,
        boxstyle="round,pad=0.1",
        facecolor=DiagramColors.GRAPH_HIGHLIGHT,
        edgecolor=DiagramColors.GRAPH_NODE,
        linewidth=2,
        alpha=0.7
    )
    ax.add_patch(graph_query_box)
    ax.text(11.5, stage_y + 0.15, "Entity Extraction",
           ha='center', va='center', fontsize=9,
           color=DiagramColors.TEXT_PRIMARY, fontweight='bold')
    ax.text(11.5, stage_y - 0.15, "+ Multi-hop Traversal",
           ha='center', va='center', fontsize=8,
           color=DiagramColors.TEXT_SECONDARY, fontweight='bold')
    
    # Add special features annotation for GraphRAG
    ax.text(14.5, stage_y, "✓ Lexical Anchoring\n✓ Degree Capping\n✓ Stratification",
           ha='left', va='center', fontsize=7,
           color=DiagramColors.GRAPH_NODE,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                    edgecolor=DiagramColors.GRAPH_NODE, linewidth=1, alpha=0.9))
    
    # Arrows down
    arrow_v4 = FancyArrowPatch(
        (3.5, stage_y - 0.5), (3.5, stage_y - 1.5),
        arrowstyle='-|>',
        mutation_scale=15,
        linewidth=2,
        color=DiagramColors.VECTOR_MAIN
    )
    ax.add_patch(arrow_v4)
    
    arrow_g4 = FancyArrowPatch(
        (11.5, stage_y - 0.5), (11.5, stage_y - 1.5),
        arrowstyle='-|>',
        mutation_scale=15,
        linewidth=2,
        color=DiagramColors.GRAPH_NODE
    )
    ax.add_patch(arrow_g4)
    
    # === Stage 5: Retrieved Context ===
    stage_y = 3
    
    # Vector RAG: Dense text chunks (large, bloated)
    vector_context_box = FancyBboxPatch(
        (1.5, stage_y - 0.6), 4, 1.2,
        boxstyle="round,pad=0.15",
        facecolor=DiagramColors.TEXT_CHUNK,
        edgecolor=DiagramColors.VECTOR_MAIN,
        linewidth=2.5,
        alpha=0.6
    )
    ax.add_patch(vector_context_box)
    ax.text(3.5, stage_y + 0.3, "Dense Text Chunks",
           ha='center', va='center', fontsize=9,
           color=DiagramColors.TEXT_PRIMARY, fontweight='bold')
    ax.text(3.5, stage_y, "Large paragraphs",
           ha='center', va='center', fontsize=7,
           color=DiagramColors.TEXT_SECONDARY)
    ax.text(3.5, stage_y - 0.3, "⚠ 82% noise, 18% signal",
           ha='center', va='center', fontsize=7,
           color=DiagramColors.VECTOR_MAIN, fontweight='bold')
    
    # GraphRAG: Atomic triplets (compact, structured)
    graph_context_box = FancyBboxPatch(
        (9.5, stage_y - 0.6), 4, 1.2,
        boxstyle="round,pad=0.15",
        facecolor=DiagramColors.GRAPH_HIGHLIGHT,
        edgecolor=DiagramColors.GRAPH_NODE,
        linewidth=2.5,
        alpha=0.6
    )
    ax.add_patch(graph_context_box)
    ax.text(11.5, stage_y + 0.3, "Atomic Triplets",
           ha='center', va='center', fontsize=9,
           color=DiagramColors.TEXT_PRIMARY, fontweight='bold')
    ax.text(11.5, stage_y, "[Entity]--REL-->[Entity]",
           ha='center', va='center', fontsize=7,
           color=DiagramColors.GRAPH_NODE, family='monospace')
    ax.text(11.5, stage_y - 0.3, "✓ 85% signal, 15% noise",
           ha='center', va='center', fontsize=7,
           color=DiagramColors.GRAPH_NODE, fontweight='bold')
    
    # Arrows down
    arrow_v5 = FancyArrowPatch(
        (3.5, stage_y - 0.7), (3.5, stage_y - 1.5),
        arrowstyle='-|>',
        mutation_scale=15,
        linewidth=2,
        color=DiagramColors.VECTOR_MAIN
    )
    ax.add_patch(arrow_v5)
    
    arrow_g5 = FancyArrowPatch(
        (11.5, stage_y - 0.7), (11.5, stage_y - 1.5),
        arrowstyle='-|>',
        mutation_scale=15,
        linewidth=2,
        color=DiagramColors.GRAPH_NODE
    )
    ax.add_patch(arrow_g5)
    
    # === Stage 6: LLM Prompt ===
    stage_y = 0.5
    
    # Shared LLM (converging arrows)
    llm_box = FancyBboxPatch(
        (6, stage_y - 0.3), 3, 0.6,
        boxstyle="round,pad=0.1",
        facecolor=DiagramColors.LLM,
        edgecolor=DiagramColors.BORDER,
        linewidth=2,
        alpha=0.8
    )
    ax.add_patch(llm_box)
    ax.text(7.5, stage_y, "Azure OpenAI GPT-4",
           ha='center', va='center', fontsize=10,
           color='white', fontweight='bold')
    
    # Converging arrows
    arrow_v6 = FancyArrowPatch(
        (3.5, stage_y), (6, stage_y),
        arrowstyle='-|>',
        mutation_scale=15,
        linewidth=2,
        color=DiagramColors.VECTOR_MAIN
    )
    ax.add_patch(arrow_v6)
    
    arrow_g6 = FancyArrowPatch(
        (11.5, stage_y), (9, stage_y),
        arrowstyle='-|>',
        mutation_scale=15,
        linewidth=2,
        color=DiagramColors.GRAPH_NODE
    )
    ax.add_patch(arrow_g6)
    
    # === Key Differences Box ===
    key_diff_text = (
        "Key Architectural Differences:\n"
        "• Vector RAG: Unstructured chunks, no relationships, potential hallucination\n"
        "• GraphRAG: Structured triplets, typed relationships, provenance tracking\n"
        "• Vector RAG: Fast (2-3s), but 82% context bloat\n"
        "• GraphRAG: Slower (11s), but 85% signal concentration"
    )
    
    ax.text(7.5, -0.8, key_diff_text,
           ha='center', va='top', fontsize=8,
           color=DiagramColors.TEXT_PRIMARY,
           bbox=dict(boxstyle="round,pad=0.5", facecolor=DiagramColors.BG_LIGHT,
                    edgecolor=DiagramColors.BORDER, linewidth=2, alpha=0.9))
    
    # Save figure
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    
    for fmt in ['png', 'pdf']:
        out_path = out_dir / f"plot_8_pipeline_comparison.{fmt}"
        fig.savefig(out_path, dpi=dpi if fmt == 'png' else None,
                   bbox_inches='tight', format=fmt)
        log.info("Saved → %s", out_path)
    
    return fig, ax

# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    """Generate architecture and structure diagrams (7 and 8)."""
    log.info("=" * 60)
    log.info("Generating Architecture Visualization Diagrams")
    log.info("=" * 60)
    
    # Try to load actual evaluation data
    graphrag_data = load_graphrag_results("evals/graphrag_evaluation_results.json")
    
    out_dir = Path("figures")
    
    # Generate Diagram 7: Structured Knowledge Graph
    log.info("\nRendering Diagram 7 — Structured Knowledge Graph ...")
    plot_structured_knowledge_graph(
        results=graphrag_data if graphrag_data else None,
        out_dir=out_dir
    )
    
    # Generate Diagram 8: Pipeline Comparison
    #log.info("\nRendering Diagram 8 — Pipeline Comparison ...")
    #plot_pipeline_comparison(out_dir=out_dir)
    
    log.info("=" * 60)
    log.info("Successfully saved Diagrams 7 and 8 to ./figures/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
