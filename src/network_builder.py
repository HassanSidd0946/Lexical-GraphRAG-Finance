"""
Phase 3.5: Global Graph Optimizer (NetworkX)
==============================================
Mathematically optimizes the raw LLM Knowledge Graph.
- Performs Alias-Aware Fuzzy Entity Resolution.
- Calculates PageRank Centrality.
- Blends normalized LLM weights with structural centrality.
- Preserves provenance by concatenating edge explanations.
- Prunes isolated noise.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
from difflib import SequenceMatcher
import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_python

# ──────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
INPUT_FILE = Path("phase3_deduplicated_graph.json")
OUTPUT_FILE = Path("phase3_5_final_knowledge_graph.json")
FUZZY_THRESHOLD = 0.85  # 85% string similarity to trigger a merge

def string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# ──────────────────────────────────────────────
# 1. Alias-Aware Fuzzy Entity Resolution
# ──────────────────────────────────────────────
def resolve_entities(nodes: List[dict], edges: List[dict]) -> tuple[List[dict], List[dict]]:
    logger.info("Starting Alias-Aware Fuzzy Entity Resolution...")
    
    nodes_by_type: Dict[str, List[dict]] = {}
    for node in nodes:
        if node.get("type") == "Chunk":
            continue
        nodes_by_type.setdefault(node["type"], []).append(node)
        
    canonical_mapping: Dict[str, str] = {}
    merged_nodes: Dict[str, dict] = {}
    merge_count = 0

    for node_type, type_nodes in nodes_by_type.items():
        # Sort by length: shortest name becomes the canonical base
        type_nodes.sort(key=lambda x: len(x["name"]))
        
        for node in type_nodes:
            node_name = node["name"]
            matched_canonical = None
            
            for canonical_name, canonical_data in merged_nodes.items():
                if canonical_data["type"] != node_type:
                    continue
                
                # Check direct similarity
                if string_similarity(node_name, canonical_name) >= FUZZY_THRESHOLD:
                    matched_canonical = canonical_name
                    break
                
                # Check against aliases
                for alias in canonical_data.get("aliases", []):
                    if string_similarity(node_name, alias) >= FUZZY_THRESHOLD:
                        matched_canonical = canonical_name
                        break
            
            if matched_canonical:
                merge_count += 1
                canonical_mapping[node_name] = matched_canonical
                
                # Safely merge descriptions
                existing_desc = merged_nodes[matched_canonical].get("description", "")
                incoming_desc = node.get("description", "")
                if incoming_desc and incoming_desc not in existing_desc:
                    merged_nodes[matched_canonical]["description"] = f"{existing_desc} | {incoming_desc}"
                
                # Merge source chunks
                existing_chunks = set(merged_nodes[matched_canonical].get("source_chunks", []))
                existing_chunks.update(node.get("source_chunks", []))
                merged_nodes[matched_canonical]["source_chunks"] = list(existing_chunks)
                
                # Add discarded name to aliases
                if node_name not in merged_nodes[matched_canonical]["aliases"]:
                    merged_nodes[matched_canonical]["aliases"].append(node_name)
            else:
                canonical_mapping[node_name] = node_name
                merged_nodes[node_name] = node.copy()

    # Restore Chunk nodes
    for node in nodes:
        if node.get("type") == "Chunk":
            merged_nodes[node["name"]] = node
            canonical_mapping[node["name"]] = node["name"]

    logger.info(f"Resolution Complete: Merged {merge_count} semantic duplicate nodes.")

    # Remap edges to canonical names
    updated_edges = []
    for edge in edges:
        new_source = canonical_mapping.get(edge["source"])
        new_target = canonical_mapping.get(edge["target"])
        if new_source and new_target:
            edge["source"] = new_source
            edge["target"] = new_target
            updated_edges.append(edge)

    return list(merged_nodes.values()), updated_edges

# ──────────────────────────────────────────────
# 2. NetworkX Graph Building & Math
# ──────────────────────────────────────────────
def optimize_graph(nodes: List[dict], edges: List[dict]) -> dict:
    logger.info("Building NetworkX Directed Graph...")
    G = nx.DiGraph()

    for node in nodes:
        G.add_node(node["name"], **node)

    consolidation_count = 0

    # Consolidate Edges (Preserving Provenance)
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        
        if G.has_edge(source, target):
            consolidation_count += 1
            existing_weight = G[source][target].get("weight", 1)
            existing_exp = G[source][target].get("explanation", "")
            incoming_exp = edge.get("explanation", "")
            
            # Concatenate unique explanations
            if incoming_exp and incoming_exp not in existing_exp:
                G[source][target]["explanation"] = f"{existing_exp} | {incoming_exp}"
                
            # Keep the highest structural weight
            if edge.get("weight", 1) > existing_weight:
                G[source][target]["weight"] = edge["weight"]
                G[source][target]["relation"] = edge["relation"]
        else:
            G.add_edge(source, target, **edge)

    logger.info(f"Consolidated {consolidation_count} redundant edges into multi-provenance edges.")

    # Calculate PageRank
    logger.info("Calculating PageRank Centrality...")
    try:
        pagerank_scores = nx.pagerank(G, weight='weight', alpha=0.85)
    except ModuleNotFoundError as exc:
        if "scipy" not in str(exc).lower():
            raise
        logger.warning("scipy not installed; falling back to pure-Python PageRank.")
        pagerank_scores = _pagerank_python(G, weight='weight', alpha=0.85)
    max_pr = max(pagerank_scores.values()) if pagerank_scores else 1
    
    for node_name in G.nodes():
        raw_score = pagerank_scores.get(node_name, 0)
        normalized_score = round((raw_score / max_pr) * 100, 2)
        G.nodes[node_name]["centrality_score"] = normalized_score

    # Calculate Blended Retrieval Score
    logger.info("Calculating Balanced Retrieval Scores (40% LLM / 60% Centrality)...")
    for u, v, data in G.edges(data=True):
        if data.get("relation") == "MENTIONED_IN":
            data["retrieval_score"] = 100.0 
        else:
            u_score = G.nodes[u].get("centrality_score", 1.0)
            v_score = G.nodes[v].get("centrality_score", 1.0)
            llm_weight = data.get("weight", 1)
            
            # Normalize 1-5 scale to 20-100 scale
            normalized_llm = (llm_weight / 5.0) * 100.0
            
            retrieval_score = (normalized_llm * 0.4) + (((u_score + v_score) / 2) * 0.6)
            data["retrieval_score"] = round(retrieval_score, 2)

    # Orphan Pruning
    orphans = [node for node, degree in dict(G.degree()).items() if degree == 0]
    G.remove_nodes_from(orphans)
    if orphans:
        logger.info(f"Pruned {len(orphans)} orphaned nodes from the final graph.")

    # Export to Dictionary
    final_nodes = [{"name": n, **d} for n, d in G.nodes(data=True)]
    final_edges = [{"source": u, "target": v, **d} for u, v, d in G.edges(data=True)]

    return {
        "nodes": final_nodes,
        "edges": final_edges,
        "stats": {
            "final_node_count": G.number_of_nodes(),
            "final_edge_count": G.number_of_edges()
        }
    }

# ──────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────
def main():
    if not INPUT_FILE.exists():
        logger.error(f"Could not find {INPUT_FILE}. Run Phase 3 extraction first.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    resolved_nodes, resolved_edges = resolve_entities(raw_data["nodes"], raw_data["edges"])
    final_graph_data = optimize_graph(resolved_nodes, resolved_edges)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_graph_data, f, indent=2)

    logger.info(f"SUCCESS! Final Graph saved to {OUTPUT_FILE}")
    logger.info(f"Final Nodes: {final_graph_data['stats']['final_node_count']}")
    logger.info(f"Final Edges: {final_graph_data['stats']['final_edge_count']}")

if __name__ == "__main__":
    main()