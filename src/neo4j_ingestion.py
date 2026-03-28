"""
Phase 4.1: Production-Grade Neo4j Graph Ingestion
===================================================
Loads the Knowledge Graph JSON into a Neo4j database.
"""

import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from neo4j import GraphDatabase, exceptions

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
@dataclass
class Config:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "password")
    input_file: Path = Path("phase3_5_final_knowledge_graph.json")
    batch_size: int = int(os.getenv("NEO4J_BATCH_SIZE", "500"))


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def chunker(seq: list, size: int):
    for pos in range(0, len(seq), size):
        yield seq[pos : pos + size]


# Whitelist: only allow alphanumeric + underscore in dynamic Cypher labels/rel-types
# to prevent Cypher injection via malformed node types in the source data.
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def sanitize_identifier(value: str, fallback: str) -> str:
    clean = re.sub(r"\s+", "_", value.strip())
    return clean if _SAFE_IDENTIFIER.match(clean) else fallback


# ──────────────────────────────────────────────
# Ingestor
# ──────────────────────────────────────────────
class GraphIngestor:
    def __init__(self, cfg: Config):
        self.driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
        self.batch_size = cfg.batch_size
        self.metrics = defaultdict(int)  # nodes/edges: attempted, ingested, skipped, failed

    # Context manager so callers can use `with GraphIngestor(cfg) as g:`
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.driver.close()

    # ── Setup ──────────────────────────────────
    def verify_connection(self):
        logger.info("Verifying Neo4j connectivity...")
        try:
            self.driver.verify_connectivity()
            logger.info("Connection established.")
        except Exception as e:
            logger.error(f"Cannot reach Neo4j: {e}")
            raise

    def clear_database(self):
        logger.warning("Clearing existing Neo4j database...")
        with self.driver.session() as session:
            session.run(
                "MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS"
            )

    def create_schema(self):
        """Unique constraint + index on Entity.name for fast MATCH during edge ingestion."""
        logger.info("Creating schema (constraint + index)...")
        with self.driver.session() as session:
            try:
                session.run(
                    "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                    "FOR (n:Entity) REQUIRE n.name IS UNIQUE"
                )
                # Range index accelerates the MATCH lookups in _merge_edges_tx
                session.run(
                    "CREATE INDEX entity_name_idx IF NOT EXISTS "
                    "FOR (n:Entity) ON (n.name)"
                )
            except exceptions.ClientError as e:
                logger.debug(f"Schema already exists (safe to ignore): {e}")

    # ── Node Ingestion ─────────────────────────
    def _merge_nodes_tx(self, tx, node_type: str, batch: list):
        query = f"""
        UNWIND $batch AS node
        MERGE (n:`{node_type}` {{name: node.name}})
        SET n:Entity, n += node.properties
        """
        tx.run(query, batch=batch)

    def ingest_nodes(self, nodes: list):
        self.metrics["nodes_attempted"] = len(nodes)
        logger.info(f"Ingesting {len(nodes)} nodes...")

        by_type: dict = defaultdict(list)
        for node in nodes:
            if not isinstance(node.get("name"), str) or not node["name"].strip():
                logger.warning(f"Skipping malformed node: {node}")
                self.metrics["nodes_skipped"] += 1
                continue

            raw_type = node.get("type", "Entity")
            node_type = sanitize_identifier(raw_type, "Entity")
            props = {k: v for k, v in node.items() if k not in ("name", "type")}
            by_type[node_type].append({"name": node["name"], "properties": props})

        self._run_batched(by_type, self._merge_nodes_tx, "nodes")

    # ── Edge Ingestion ─────────────────────────
    def _merge_edges_tx(self, tx, rel_type: str, batch: list):
        query = f"""
        UNWIND $batch AS edge
        MATCH (s:Entity {{name: edge.source}})
        MATCH (t:Entity {{name: edge.target}})
        MERGE (s)-[r:`{rel_type}`]->(t)
        SET r += edge.properties
        """
        tx.run(query, batch=batch)

    def ingest_edges(self, edges: list):
        # Deduplicate edges before hitting the DB (same source/target/relation)
        seen: set = set()
        unique_edges = []
        for edge in edges:
            key = (edge.get("source"), edge.get("target"), edge.get("relation"))
            if key not in seen:
                seen.add(key)
                unique_edges.append(edge)

        duplicates = len(edges) - len(unique_edges)
        if duplicates:
            logger.info(f"Dropped {duplicates} duplicate edges before ingestion.")

        self.metrics["edges_attempted"] = len(unique_edges)
        logger.info(f"Ingesting {len(unique_edges)} edges...")

        by_rel: dict = defaultdict(list)
        for edge in unique_edges:
            if not edge.get("source") or not edge.get("target"):
                logger.warning(f"Skipping malformed edge: {edge}")
                self.metrics["edges_skipped"] += 1
                continue

            raw_rel = edge.get("relation", "RELATED_TO")
            rel_type = sanitize_identifier(raw_rel, "RELATED_TO")
            props = {k: v for k, v in edge.items() if k not in ("source", "target", "relation")}
            by_rel[rel_type].append({
                "source": edge["source"],
                "target": edge["target"],
                "properties": props,
            })

        self._run_batched(by_rel, self._merge_edges_tx, "edges")

    # ── Shared Batch Runner ────────────────────
    def _run_batched(self, by_key: dict, tx_fn, entity: str):
        """Generic batch executor with 1-by-1 fallback on failure."""
        with self.driver.session() as session:
            for key, items in by_key.items():
                for batch in chunker(items, self.batch_size):
                    try:
                        session.execute_write(tx_fn, key, batch)
                        self.metrics[f"{entity}_ingested"] += len(batch)
                    except Exception as e:
                        logger.warning(
                            f"Batch failed for '{key}' — falling back to 1-by-1. Error: {e}"
                        )
                        for item in batch:
                            try:
                                session.execute_write(tx_fn, key, [item])
                                self.metrics[f"{entity}_ingested"] += 1
                            except Exception as inner_e:
                                logger.error(f"Failed single {entity[:-1]}: {item} — {inner_e}")
                                self.metrics[f"{entity}_failed"] += 1

    # ── Audit Summary ──────────────────────────
    def print_audit_summary(self):
        m = self.metrics
        logger.info("=== INGESTION AUDIT SUMMARY ===")
        for entity in ("nodes", "edges"):
            logger.info(
                f"{entity.capitalize():6s} — "
                f"Attempted: {m[f'{entity}_attempted']:>5} | "
                f"Ingested: {m[f'{entity}_ingested']:>5} | "
                f"Skipped: {m[f'{entity}_skipped']:>4} | "
                f"Failed: {m[f'{entity}_failed']:>4}"
            )
        logger.info("===============================")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    cfg = Config()  # Edit Config fields above, or override here

    try:
        graph_data = json.loads(cfg.input_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error(f"Input file not found: {cfg.input_file}")
        return

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    start = time.perf_counter()

    try:
        with GraphIngestor(cfg) as g:
            g.verify_connection()
            g.clear_database()
            g.create_schema()
            g.ingest_nodes(nodes)
            g.ingest_edges(edges)
            logger.info(f"Done in {time.perf_counter() - start:.2f}s")
            g.print_audit_summary()
    except Exception as e:
        logger.error(f"Fatal error: {e}")


if __name__ == "__main__":
    main()