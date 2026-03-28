"""
SEC 10-K Knowledge Graph Extractor (Phase 3)
==============================================
Optimized pipeline with async processing, strict bounded ontology, 
programmatic sanity/validation layers, graph deduplication, 
edge weighting, provenance tracking, and robust error handling.
"""

import json
import logging
import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional
from pathlib import Path
from collections import defaultdict

import aiofiles
from tqdm.asyncio import tqdm as async_tqdm
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIConnectionError
from dotenv import load_dotenv
import os

# ──────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("graph_extractor.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def _build_chat_model(cfg: "ExtractorConfig"):
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_deployment = (
        os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        or os.getenv("AZURE_OPENAI_MODEL_NAME")
    )

    if azure_endpoint and azure_api_key and azure_api_version and azure_deployment:
        logger.info("Using Azure OpenAI chat deployment '%s'.", azure_deployment)

        llm_kwargs = {
            "azure_endpoint": azure_endpoint,
            "api_key": azure_api_key,
            "openai_api_version": azure_api_version,
            "azure_deployment": azure_deployment,
        }

        if not azure_deployment.lower().startswith("o"):
            llm_kwargs["temperature"] = cfg.temperature
        else:
            logger.info(
                "Skipping explicit temperature for deployment '%s' (uses model default).",
                azure_deployment,
            )

        return AzureChatOpenAI(
            **llm_kwargs,
        )

    missing = []
    if not azure_endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not azure_api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if not azure_api_version:
        missing.append("AZURE_OPENAI_API_VERSION")
    if not azure_deployment:
        missing.append(
            "one of: AZURE_OPENAI_CHAT_DEPLOYMENT_NAME / AZURE_OPENAI_DEPLOYMENT_NAME / AZURE_OPENAI_MODEL_NAME"
        )

    raise EnvironmentError(
        "Missing Azure OpenAI credentials: " + ", ".join(missing)
    )


# ──────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────
@dataclass
class ExtractorConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_concurrency: int = 10          # Protects against RateLimitError
    max_retries: int = 4
    retry_min_wait: float = 2.0
    retry_max_wait: float = 30.0
    debug_limit: Optional[int] = 3     # SET TO None TO PROCESS ALL 75 CHUNKS
    input_file: Path = field(default_factory=lambda: Path("sec_semantic_chunks_master.jsonl"))
    output_file: Path = field(default_factory=lambda: Path("phase3_extracted_graph.jsonl"))
    deduplicated_output: Path = field(default_factory=lambda: Path("phase3_deduplicated_graph.json"))
    failed_chunks_file: Path = field(default_factory=lambda: Path("failed_chunks.jsonl"))


# ──────────────────────────────────────────────
# 2. Strict Pydantic Schema
# ──────────────────────────────────────────────
NodeType = Literal[
    "Company", "RiskFactor", "Regulation",
    "FinancialMetric", "BusinessUnit", "ExternalEntity",
    "FinancialInstrument", "Obligation", "Event"
]

EdgeType = Literal[
    "EXPOSES_TO", "AFFECTS", "CAUSES", "MITIGATES", 
    "GOVERNS", "DEPENDS_ON", "COMPETES_WITH", "OWNS", 
    "REGULATES", "INCREASES", "DECREASES", "ISSUES"
]

class Node(BaseModel):
    name: str = Field(..., description="Canonical, normalized entity name (e.g., 'JPMorgan Chase'). Merge all aliases.")
    type: NodeType = Field(..., description="Strict category of this node.")
    aliases: List[str] = Field(default_factory=list, description="Other names used for this entity in the text.")
    description: str = Field(..., description="Brief summary capturing nuanced context within the chunk. Max 2 sentences.")

class Edge(BaseModel):
    source: str = Field(..., description="Exact name of the source node.")
    target: str = Field(..., description="Exact name of the target node.")
    relation: EdgeType = Field(..., description="STRICT relationship verb from the allowed EdgeType list.")
    weight: int = Field(..., ge=1, le=5, description="Importance of this relationship. 1 = Minor detail, 5 = Critical risk/driver.")
    explanation: str = Field(..., description="Why this edge exists based on the text. Max 2 sentences.")

class GraphExtraction(BaseModel):
    nodes: List[Node] = Field(..., description="Unique entities extracted.")
    edges: List[Edge] = Field(..., description="Relationships between extracted nodes.")


# ──────────────────────────────────────────────
# 3. Extraction Prompt
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert financial data scientist building a Knowledge Graph from SEC 10-K filings.
Extract a highly dense, accurate property graph from the provided text chunk.

CRITICAL CONSTRAINTS (FAILING THESE WILL BREAK THE SYSTEM):
1. SIZE LIMIT: Extract a MAXIMUM of 12 highly critical nodes per chunk. Do not over-extract noise.
2. DESCRIPTION LIMIT: Keep `description` and `explanation` strictly under 2 concise sentences. No fluff.
3. NORMALIZATION: Collapse aliases to a single canonical `name`. (e.g., 'JPM' -> name: 'JPMorgan Chase', aliases: ['JPM']).
4. STRICT EDGE TYPES: You MUST ONLY use the exact verbs provided in the EdgeType schema.
5. REFERENTIAL INTEGRITY: Every `source` and `target` in your edges MUST match a `name` in your nodes list exactly.
6. WEIGHTING (1-5): Score every edge. 5 = Systemic risk/major driver. 1 = Minor operational detail.
7. EXHAUSTION: Capture secondary risks, financial instruments (debt/derivatives), and obligations, not just the primary subject.
"""

_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Extract the property graph from this SEC 10-K chunk:\n\n{text}"),
])


# ──────────────────────────────────────────────
# 4. Programmatic Sanity & Validation Layer
# ──────────────────────────────────────────────
def sanitize_and_validate(extraction: GraphExtraction, chunk_id: str) -> GraphExtraction:
    orig_node_count = len(extraction.nodes)
    orig_edge_count = len(extraction.edges)

    # 1. Lexical Normalization & Intra-chunk Deduplication
    unique_nodes = {}
    for node in extraction.nodes:
        clean_name = " ".join(node.name.strip().split()).title()
        node.name = clean_name
        
        if node.aliases:
            node.aliases = [" ".join(a.strip().split()).title() for a in node.aliases]
            
        if clean_name not in unique_nodes:
            unique_nodes[clean_name] = node
        else:
            unique_nodes[clean_name].aliases.extend(node.aliases)
            unique_nodes[clean_name].aliases = list(set(unique_nodes[clean_name].aliases))

    valid_node_names = set(unique_nodes.keys())

    # 2. Referential Integrity (Kill Ghost Edges)
    valid_edges = []
    for edge in extraction.edges:
        edge.source = " ".join(edge.source.strip().split()).title()
        edge.target = " ".join(edge.target.strip().split()).title()
        
        if edge.source in valid_node_names and edge.target in valid_node_names:
            valid_edges.append(edge)

    # 3. Hard Edge Limit (Cap runaway extraction)
    if len(valid_edges) > 30:
        logger.warning(f"[{chunk_id}] Edge explosion ({len(valid_edges)}). Truncating to 30 top-weighted edges.")
        valid_edges.sort(key=lambda x: x.weight, reverse=True)
        valid_edges = valid_edges[:30]

    # 4. Orphan Pruning (Kill Isolated Nodes)
    connected_names = set()
    for edge in valid_edges:
        connected_names.add(edge.source)
        connected_names.add(edge.target)

    final_nodes = [
        node for name, node in unique_nodes.items() 
        if name in connected_names or node.type == "Company"
    ]

    dropped_nodes = orig_node_count - len(final_nodes)
    dropped_edges = orig_edge_count - len(valid_edges)
    if dropped_nodes > 0 or dropped_edges > 0:
        logger.info(f"[{chunk_id}] Sanity Filter: Pruned {dropped_nodes} invalid nodes & {dropped_edges} invalid edges.")

    extraction.nodes = final_nodes
    extraction.edges = valid_edges
    return extraction


# ──────────────────────────────────────────────
# 5. Retry-Decorated Extractor (With Provenance)
# ──────────────────────────────────────────────
def _build_retry_decorator(cfg: ExtractorConfig):
    return retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        stop=stop_after_attempt(cfg.max_retries),
        wait=wait_exponential(min=cfg.retry_min_wait, max=cfg.retry_max_wait),
        before_sleep=lambda rs: logger.warning(
            f"Retry {rs.attempt_number}/{cfg.max_retries} after error: {rs.outcome.exception()}"
        ),
        reraise=True,
    )

async def _call_llm(chain, text: str) -> GraphExtraction:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: chain.invoke({"text": text}))

async def extract_graph_from_chunk(chunk_text: str, chunk_id: str, chain, retry_decorator) -> dict:
    @retry_decorator
    async def _invoke():
        return await _call_llm(chain, chunk_text)

    try:
        raw_extraction: GraphExtraction = await _invoke()
        extraction = sanitize_and_validate(raw_extraction, chunk_id)
    except Exception as exc:
        logger.error(f"[{chunk_id}] Extraction failed: {exc}")
        return {"chunk_id": chunk_id, "nodes": [], "edges": [], "status": "failed"}

    nodes = [n.model_dump() for n in extraction.nodes]
    edges = [e.model_dump() for e in extraction.edges]

    # PROVENANCE ANCHOR: Programmatically bind every valid entity to its source chunk
    chunk_node = {
        "name": chunk_id,
        "type": "Chunk",
        "aliases": [],
        "description": "Source text snippet from SEC 10-K.",
    }
    nodes.append(chunk_node)

    for node in extraction.nodes:
        edges.append({
            "source": node.name,
            "target": chunk_id,
            "relation": "MENTIONED_IN",
            "weight": 5, 
            "explanation": "Programmatically linked for retrieval provenance.",
        })

    return {"chunk_id": chunk_id, "nodes": nodes, "edges": edges, "status": "ok"}


# ──────────────────────────────────────────────
# 6. Graph Deduplication & Merging
# ──────────────────────────────────────────────
def build_merged_graph(raw_results: List[dict]) -> dict:
    merged_nodes: dict = {}    
    merged_edges: dict = {}    
    chunk_sources: dict = defaultdict(list)  

    for result in raw_results:
        if result["status"] == "failed":
            continue
        chunk_id = result["chunk_id"]

        for node in result["nodes"]:
            key = (node["name"], node["type"])
            if key not in merged_nodes:
                merged_nodes[key] = node.copy()
            else:
                existing_desc = merged_nodes[key]["description"]
                incoming_desc = node.get("description", "")
                if incoming_desc and incoming_desc not in existing_desc:
                    merged_nodes[key]["description"] = f"{existing_desc} | {incoming_desc}"
                
                existing_aliases = set(merged_nodes[key].get("aliases", []))
                existing_aliases.update(node.get("aliases", []))
                merged_nodes[key]["aliases"] = sorted(list(existing_aliases))

            if node["type"] != "Chunk":
                chunk_sources[node["name"]].append(chunk_id)

        for edge in result["edges"]:
            key = (edge["source"], edge["target"], edge["relation"])
            if key not in merged_edges:
                merged_edges[key] = edge.copy()
            else:
                existing_weight = merged_edges[key].get("weight", 1)
                incoming_weight = edge.get("weight", 1)
                
                if incoming_weight > existing_weight:
                    merged_edges[key]["weight"] = incoming_weight
                    merged_edges[key]["explanation"] = edge["explanation"]
                elif incoming_weight == existing_weight and len(edge["explanation"]) > len(merged_edges[key]["explanation"]):
                    merged_edges[key]["explanation"] = edge["explanation"]

    final_nodes = []
    for node in merged_nodes.values():
        if node["type"] != "Chunk":
            node["source_chunks"] = sorted(set(chunk_sources.get(node["name"], [])))
        final_nodes.append(node)

    return {
        "nodes": final_nodes,
        "edges": list(merged_edges.values()),
        "stats": {
            "total_nodes": len(final_nodes),
            "total_edges": len(merged_edges),
        },
    }


# ──────────────────────────────────────────────
# 7. Async Main Pipeline
# ──────────────────────────────────────────────
async def run_pipeline(cfg: ExtractorConfig) -> None:
    load_dotenv()

    if not cfg.input_file.exists():
        logger.error(f"Input file not found: {cfg.input_file}")
        return

    chunks = []
    with open(cfg.input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chunks.append((data["chunk_id"], data["text"]))

    if cfg.debug_limit:
        chunks = chunks[: cfg.debug_limit]
        logger.info(f"Debug mode: processing first {cfg.debug_limit} chunks.")

    logger.info(f"Loaded {len(chunks)} chunks. Starting async extraction...")

    llm = _build_chat_model(cfg)
    structured_llm = llm.with_structured_output(GraphExtraction)
    chain = _PROMPT_TEMPLATE | structured_llm
    retry_dec = _build_retry_decorator(cfg)

    semaphore = asyncio.Semaphore(cfg.max_concurrency)

    async def bounded_extract(chunk_id: str, text: str) -> dict:
        async with semaphore:
            return await extract_graph_from_chunk(text, chunk_id, chain, retry_dec)

    tasks = [bounded_extract(cid, txt) for cid, txt in chunks]
    results: List[dict] = await async_tqdm.gather(*tasks, desc="Extracting chunks")

    async with aiofiles.open(cfg.output_file, "w", encoding="utf-8") as f:
        for r in results:
            await f.write(json.dumps(r) + "\n")
    logger.info(f"Raw graph data saved -> {cfg.output_file}")

    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        async with aiofiles.open(cfg.failed_chunks_file, "w", encoding="utf-8") as f:
            for r in failed:
                await f.write(json.dumps({"chunk_id": r["chunk_id"]}) + "\n")
        logger.warning(f"{len(failed)} chunks failed. Logged -> {cfg.failed_chunks_file}")

    logger.info("Merging and deduplicating graph across all chunks...")
    merged = build_merged_graph(results)
    async with aiofiles.open(cfg.deduplicated_output, "w", encoding="utf-8") as f:
        await f.write(json.dumps(merged, indent=2))
    logger.info(
        f"Deduplicated graph saved -> {cfg.deduplicated_output} | "
        f"Nodes: {merged['stats']['total_nodes']} | "
        f"Edges: {merged['stats']['total_edges']}"
    )


# ──────────────────────────────────────────────
# 8. Entry Point
# ──────────────────────────────────────────────
def main():
    cfg = ExtractorConfig(
        model="gpt-4o-mini",
        max_concurrency=10,
        debug_limit=None,   # SET TO None AFTER VERIFYING THE FIRST 3 CHUNKS
    )

    start = time.perf_counter()
    asyncio.run(run_pipeline(cfg))
    elapsed = time.perf_counter() - start
    logger.info(f"Pipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()