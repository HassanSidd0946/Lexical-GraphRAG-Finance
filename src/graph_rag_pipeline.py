"""
GraphRAG Evaluation Pipeline
==============================
Async pipeline: entity extraction → Neo4j multi-hop retrieval → LLM generation.
"""

import json
import logging
import time
import os
import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Set

import aiofiles
import tiktoken
from neo4j import AsyncGraphDatabase
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from openai import RateLimitError, APIConnectionError
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type


def _load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if value.startswith('"') or value.startswith("'"):
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end] if end != -1 else value[1:]
        else:
            value = value.split("#", 1)[0].strip()

        os.environ.setdefault(key, value)


_load_env_file()

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
    # Neo4j
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password"))

    # Azure OpenAI
    azure_endpoint: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    azure_api_key: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""))
    deployment_name: str = "o4-mini"
    api_version: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"))
    llm_temperature: float = 1.0
    llm_max_tokens: int = 1500

    # Pipeline
    ground_truth_file: Path = Path("ground_truth.json")
    output_file: Path = Path("graphrag_evaluation_results.json")
    max_concurrent_tasks: int = 5
    max_context_tokens: int = 6000
    max_cache_size: int = 1000
    checkpoint_every: int = 10          # Save incremental results every N completions
    allowed_ontology: Set[str] = field(default_factory=set)


# ──────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────
try:
    _TOKENIZER = tiktoken.get_encoding("o200k_base")
except Exception:
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


# --- PROMPTS ---
ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("user",
     "You are a search query extractor. Extract 2 to 4 broad, fundamental keywords from the user's question to search a financial database. "
     "CRITICAL: Each item MUST be 1 or 2 words maximum. Do not extract long descriptive phrases. "
     "Return ONLY a comma-separated list. If no entities found, return 'NONE'.\n\n"
     "Question: {question}"),
])

GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("user",
    "You are an expert financial analyst. Answer using ONLY the provided Knowledge Graph Context.\n"
    "HALLUCINATION GUARDRAILS:\n"
    "1) Do NOT use outside knowledge or assumptions.\n"
    "2) If the context is missing details, ambiguous, or insufficient, state EXACTLY: 'The provided context does not contain sufficient information.'\n"
    "3) Do not invent entities, numbers, dates, or causal claims not explicitly present in context.\n"
    "4) For every financial claim or comparison, include explicit traceability to the supporting context by citing the source node names and any available chunk IDs from the provided context.\n"
    "5) Keep the response concise, evidence-grounded, and audit-ready.\n\n"
     "KNOWLEDGE GRAPH CONTEXT:\n{context}\n\n"
     "Question: {question}"),
])


# ──────────────────────────────────────────────
# Retry decorator — only retry on transient API errors,
# not on bad inputs / auth failures that will never succeed.
# ──────────────────────────────────────────────
_api_retry = retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    wait=wait_random_exponential(min=1, max=20),
    stop=stop_after_attempt(5),
)


# ──────────────────────────────────────────────
# Neo4j Retriever (async LRU cache + multi-hop)
# ──────────────────────────────────────────────
class AsyncNeo4jRetriever:
    _QUERY = """
    UNWIND $entity_list AS entity_kw
    WITH entity_kw, toLower(trim(entity_kw)) AS kw

    MATCH (n:Entity)
    WHERE (toLower(n.name) CONTAINS kw OR ANY(a IN n.aliases WHERE toLower(a) CONTAINS kw))
      AND COUNT { (n)--() } < 400

    MATCH (n)-[r]-(m:Entity)
    WHERE type(r) <> 'MENTIONED_IN'
      AND COUNT { (m)--() } < 400

    WITH entity_kw, r, n, m, kw,
         CASE 
            WHEN toLower(n.name) = kw OR toLower(m.name) = kw THEN 100.0
            WHEN ANY(a IN n.aliases WHERE toLower(a) = kw) OR ANY(a IN m.aliases WHERE toLower(a) = kw) THEN 30.0
            WHEN toLower(n.name) STARTS WITH kw OR toLower(m.name) STARTS WITH kw THEN 50.0
            ELSE 10.0 
         END AS lexical_score

    ORDER BY lexical_score DESC
    WITH entity_kw, kw, COLLECT({
        chunk_ids: [n.source_chunks, m.source_chunks],
        source: startNode(r).name, 
        relation: type(r), 
        target: endNode(r).name, 
        explanation: r.explanation, 
                score: lexical_score + log(
                    CASE
                        WHEN COALESCE(r.retrieval_score, 1.0) < 1.0 THEN 1.0
                        ELSE COALESCE(r.retrieval_score, 1.0)
                    END + 1.0
                )
    })[0..5] AS top_edges

    UNWIND top_edges AS item
    RETURN entity_kw,
           item.chunk_ids AS chunk_ids,
           item.source AS source,
           item.relation AS relation,
           item.target AS target,
           item.explanation AS explanation,
           item.score AS final_score
    ORDER BY final_score DESC 
    LIMIT 30
    """

    def __init__(self, cfg: Config):
        self.driver = AsyncGraphDatabase.driver(
            cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password)
        )
        self.max_context_tokens = cfg.max_context_tokens
        self.max_cache_size = cfg.max_cache_size
        self._cache: OrderedDict = OrderedDict()
        self._cache_lock = asyncio.Lock()

    async def close(self):
        await self.driver.close()

    async def get_graph_context(self, entities: List[str], question_tokens: int):
        if not entities:
            return "No relevant graph context found.", []

        uncached = []
        async with self._cache_lock:
            for e in entities:
                if e in self._cache:
                    self._cache.move_to_end(e)
                else:
                    uncached.append(e)

        if uncached:
            try:
                async with self.driver.session(database="neo4j") as session:
                    results = await session.run(self._QUERY, entity_list=uncached)
                    records = await results.data()

                # Step A: Parse the records AND grab the keyword
                processed_results = []
                for record in records:
                    processed_results.append({
                        "keyword": record["entity_kw"],
                        "chunk_ids": record["chunk_ids"],
                        "source": record["source"],
                        "relation": record["relation"],
                        "target": record["target"],
                        "explanation": record["explanation"],
                        "score": record["final_score"],
                    })

                # Step B: Map the edges ONLY to the keyword that triggered them
                temp: Dict[str, list] = {e: [] for e in uncached}
                for res in processed_results:
                    for e in uncached:
                        if e.strip().lower() == res["keyword"]:
                            temp[e].append(res)

                async with self._cache_lock:
                    for e, rels in temp.items():
                        self._cache[e] = rels
                        if len(self._cache) > self.max_cache_size:
                            self._cache.popitem(last=False)

            except Exception as e:
                logger.error(f"Neo4j query error for {uncached}: {e}")
                return "Error retrieving context.", []

        all_rels = []
        async with self._cache_lock:
            for e in entities:
                all_rels.extend(self._cache.get(e, []))

        # Deduplicate and rank
        unique_rels = {
            f"{r['source']}-{r['relation']}-{r['target']}": r for r in all_rels
        }
        sorted_rels = sorted(
            unique_rels.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        raw_ids = []
        for r in sorted_rels:
            c_ids = r.get("chunk_ids")
            if not c_ids:
                continue

            for item in c_ids:
                if isinstance(item, list):
                    raw_ids.extend(item)
                elif isinstance(item, str) and item.strip():
                    raw_ids.append(item.strip())

        retrieved_chunk_ids = list(dict.fromkeys(raw_ids))

        available_tokens = self.max_context_tokens - question_tokens - 500  # 500-token buffer
        formatted_context = self._truncate_context(sorted_rels, available_tokens)
        return formatted_context, retrieved_chunk_ids

    def _truncate_context(self, rels: List[Dict], max_tokens: int) -> str:
        parts, used = [], 0
        for r in rels:
            line = (
                f"[{r['source']}] --{r['relation']}--> [{r['target']}]. "
                f"Context: {r['explanation']} (Score: {r['score']})"
            )
            tokens = count_tokens(line)
            if used + tokens > max_tokens:
                break
            parts.append(line)
            used += tokens
        return "\n".join(parts) if parts else "No relevant graph context found."


# ──────────────────────────────────────────────
# Entity helpers
# ──────────────────────────────────────────────
def _validate_entities(entities: List[str], allowed_ontology: Set[str]) -> List[str]:
    valid = []
    for e in entities:
        cleaned = e.strip().lower()
        if len(cleaned) <= 2:
            continue
        if allowed_ontology and cleaned not in allowed_ontology:
            continue
        valid.append(cleaned)
    return valid


@_api_retry
async def _extract_entities(llm, question: str, allowed_ontology: Set[str]) -> List[str]:
    chain = ENTITY_EXTRACTION_PROMPT | llm
    response = await chain.ainvoke({"question": question})
    content = response.content.strip()
    if not content or content.upper() == "NONE":
        return []

    raw_entities = [e.strip() for e in content.split(",")]
    filtered_entities = [e for e in raw_entities if len(e) > 2]

    return _validate_entities(filtered_entities, allowed_ontology)


@_api_retry
async def _generate_answer(llm, question: str, context: str) -> str:
    chain = GENERATION_PROMPT | llm
    response = await chain.ainvoke({"context": context, "question": question})
    return response.content.strip()


# ──────────────────────────────────────────────
# Async file save (non-blocking)
# ──────────────────────────────────────────────
_save_lock = asyncio.Lock()

async def save_results(results: List[Dict], path: Path):
    async with _save_lock:
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(results, indent=2))


# ──────────────────────────────────────────────
# Per-question worker
# ──────────────────────────────────────────────
async def process_question(
    item: Dict,
    index: int,
    total: int,
    retriever: AsyncNeo4jRetriever,
    llm: AzureChatOpenAI,
    semaphore: asyncio.Semaphore,
    cfg: Config,
) -> Dict:
    async with semaphore:
        start = time.perf_counter()
        question = item.get("query", "")
        expected_chunks = item.get("expected_chunk_ids", [])
        q_type = item.get("type", "unknown")
        difficulty = item.get("difficulty", "medium")

        logger.info(f"Q{index + 1}/{total} [{q_type}]: {question[:80]}...")

        try:
            q_tokens = count_tokens(question)
            entities = await _extract_entities(llm, question, cfg.allowed_ontology)
            context, retrieved_chunks = await retriever.get_graph_context(entities, q_tokens)
            answer = await _generate_answer(llm, question, context)

            return {
                "question": question,
                "type": q_type,
                "difficulty": difficulty,
                "expected_chunk_ids": expected_chunks,
                "extracted_entities": entities,
                "retrieved_chunk_ids": retrieved_chunks,
                "retrieved_context": context,
                "generated_answer": answer,
                "latency_sec": round(time.perf_counter() - start, 2),
            }
        except Exception as e:
            logger.error(f"Failed Q{index + 1}: {e}", exc_info=True)
            return {
                "question": question,
                "error": str(e),
                "type": q_type,
                "latency_sec": round(time.perf_counter() - start, 2),
            }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
async def main_async():
    cfg = Config()

    if not cfg.ground_truth_file.exists():
        logger.error(f"Ground truth file not found: {cfg.ground_truth_file}")
        return

    qa_dataset: List[Dict] = json.loads(cfg.ground_truth_file.read_text(encoding="utf-8"))
    total = len(qa_dataset)
    logger.info(f"Loaded {total} questions.")

    llm = AzureChatOpenAI(
        azure_endpoint=cfg.azure_endpoint,
        api_key=cfg.azure_api_key,
        api_version=cfg.api_version,
        azure_deployment=cfg.deployment_name,
        temperature=cfg.llm_temperature,
        max_retries=12,
    )

    retriever = AsyncNeo4jRetriever(cfg)
    cfg.max_concurrent_tasks = 3
    semaphore = asyncio.Semaphore(cfg.max_concurrent_tasks)

    tasks = [
        process_question(item, i, total, retriever, llm, semaphore, cfg)
        for i, item in enumerate(qa_dataset)
    ]

    results: List[Dict] = []
    pipeline_start = time.perf_counter()

    for future in asyncio.as_completed(tasks):
        result = await future
        results.append(result)

        if len(results) % cfg.checkpoint_every == 0:
            await save_results(results, cfg.output_file)
            logger.info(f"Checkpoint: {len(results)}/{total} saved.")

    # Final save (only if not already on a checkpoint boundary)
    await save_results(results, cfg.output_file)
    await retriever.close()

    # Summary
    errors = sum(1 for r in results if "error" in r)
    avg_latency = (sum(r.get("latency_sec", 0) for r in results) / total) if total else 0
    elapsed = time.perf_counter() - pipeline_start
    logger.info(
        f"Done in {elapsed:.2f}s — "
        f"{total - errors}/{total} succeeded, {errors} failed. "
        f"Results → {cfg.output_file}"
    )
    logger.info(
        f"DONE: {total - errors}/{total} succeeded. "
        f"Avg Latency: {avg_latency:.2f}s. "
        f"Results saved to: {cfg.output_file}"
    )


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()