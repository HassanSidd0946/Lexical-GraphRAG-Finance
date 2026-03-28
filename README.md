# Knowledge Graph + GraphRAG for SEC 10-K Analysis

This repository implements an end-to-end research pipeline for financial-document intelligence:

- SEC 10-K section extraction and semantic chunking
- LLM-based knowledge graph extraction and optimization
- Neo4j graph ingestion and GraphRAG retrieval-generation
- Hybrid dense+sparse retrieval baseline (Vector/BM25 fusion)
- Retrieval evaluation and publication-style visualizations

The project is currently structured as a research codebase (script-first), optimized for reproducible experiments rather than packaging.

## Repository Structure

```text
.
├── src/
│   ├── fetch_data.py
│   ├── create_semantic_chunks.py
│   ├── knowledge_graph_extractor.py
│   ├── network_builder.py
│   ├── neo4j_ingestion.py
│   ├── graph_rag_pipeline.py
│   ├── hybrid_engine.py
│   └── evaluation_metric_knowledge_graph.py
├── data/
│   ├── ground_truth.json
│   ├── sec_semantic_chunks_master.jsonl
│   ├── phase3_extracted_graph.jsonl
│   ├── phase3_deduplicated_graph.json
│   ├── phase3_5_final_knowledge_graph.json
│   └── sec_data/
├── evals/
│   ├── evaluation_metric.py
│   ├── graphrag_evaluation_results.json
│   └── eval_results/
├── figures/
├── logs/
├── chroma_db/
└── visualization.py
```

## What This Project Solves

- Improves retrieval traceability by mapping answers to chunk-level provenance (`retrieved_chunk_ids`)
- Compares GraphRAG behavior against a hybrid vector+BM25 baseline
- Provides metrics for ranking quality (MRR, Precision@k, Recall@k, NDCG@k)
- Produces thesis/paper-ready figures for architecture comparison

## Prerequisites

- Python 3.10+
- Neo4j (local or remote), Bolt enabled
- Azure OpenAI resources:
	- Chat deployment (used by graph extraction / generation)
	- Embedding deployment (used by hybrid engine)

## Environment Setup

### 1) Create / activate virtual environment

PowerShell (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install dependencies

Install from the committed dependency file:

```powershell
pip install -r requirements.txt
```

### 3) Configure `.env`

Create/update `.env` in repository root:

```env
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Chat deployment (used by extractor; GraphRAG currently defaults to "o4-mini")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=o4-mini

# Embedding deployment (used by hybrid engine)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_BATCH_SIZE=500
```

## End-to-End Workflow

> Important: several scripts use relative paths. Run commands from repo root unless noted.

### Phase 1 — Fetch SEC data

```powershell
python src/fetch_data.py
```

Output: raw section text + metadata under `data/sec_data/` (or `sec_data/`, depending on your run context).

### Phase 2 — Create semantic chunks

```powershell
python src/create_semantic_chunks.py
```

Output: `sec_semantic_chunks_master.jsonl` (chunk corpus).

### Phase 3 — Extract knowledge graph from chunks

```powershell
python src/knowledge_graph_extractor.py
```

Outputs:

- `phase3_extracted_graph.jsonl`
- `phase3_deduplicated_graph.json`
- `failed_chunks.jsonl` (if extraction failures occur)

### Phase 3.5 — Optimize graph globally

```powershell
python src/network_builder.py
```

Output: `phase3_5_final_knowledge_graph.json`

### Phase 4.1 — Ingest graph into Neo4j

```powershell
python src/neo4j_ingestion.py
```

Effect: clears DB, creates schema, ingests nodes/edges from `phase3_5_final_knowledge_graph.json`.

### Phase 5 — Run GraphRAG evaluation pipeline

`src/graph_rag_pipeline.py` expects `ground_truth.json` in current working directory.

If your ground truth is in `data/ground_truth.json`, copy it once:

```powershell
Copy-Item data/ground_truth.json ground_truth.json -Force
python src/graph_rag_pipeline.py
```

Output: `graphrag_evaluation_results.json`

Then run GraphRAG metric tables:

```powershell
python src/evaluation_metric_knowledge_graph.py
```

### Baseline — Evaluate Hybrid Vector+BM25 retrieval

`evals/evaluation_metric.py` imports `hybrid_engine` from `src/`, so set `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "src"
python evals/evaluation_metric.py `
	--data-file data/sec_semantic_chunks_master.jsonl `
	--ground-truth data/ground_truth.json `
	--chroma-dir chroma_db `
	--alpha 0.7 `
	--top-k 5 `
	--output-dir evals/eval_results
```

Outputs:

- `evals/eval_results/metrics_summary.json`
- `evals/eval_results/metrics_per_query.jsonl`

### Visualization — Generate publication figures

`visualization.py` reads from current working directory:

- `graphrag_evaluation_results.json`
- `metrics_per_query.jsonl`

Quick run from root:

```powershell
Copy-Item evals/graphrag_evaluation_results.json graphrag_evaluation_results.json -Force
Copy-Item evals/eval_results/metrics_per_query.jsonl metrics_per_query.jsonl -Force
python visualization.py
```

Output: figure files in `figures/` (`.pdf` and `.png`).

## Key Scripts at a Glance

- `src/fetch_data.py`: fetches latest 10-K sections with retry/rate limiting and metadata cataloging
- `src/create_semantic_chunks.py`: token-aware semantic splitting into JSONL chunks
- `src/knowledge_graph_extractor.py`: async LLM graph extraction with schema and sanitation
- `src/network_builder.py`: deduplication + centrality-enhanced graph optimization
- `src/neo4j_ingestion.py`: production-style Neo4j ingestion with batching and audit logs
- `src/graph_rag_pipeline.py`: entity extraction + graph retrieval + answer generation + traceability
- `src/hybrid_engine.py`: dense+sparse fusion baseline retrieval engine
- `evals/evaluation_metric.py`: baseline retrieval evaluator + JSON/JSONL reports
- `src/evaluation_metric_knowledge_graph.py`: GraphRAG metric tables (global + stratified)
- `visualization.py`: GraphRAG vs baseline comparison plots

## Troubleshooting

- **`ModuleNotFoundError: hybrid_engine`**
	- Set `PYTHONPATH` to `src` before running `evals/evaluation_metric.py`.

- **Azure auth/deployment errors**
	- Verify `.env` values and deployment names.
	- Ensure API version supports your model (e.g., `o4-mini` with `2024-12-01-preview` or newer).

- **Neo4j connection failure**
	- Validate `NEO4J_URI`, credentials, and that Neo4j is running.

- **Missing file path errors**
	- Most scripts are path-relative; run from root or pass explicit CLI args where supported.

## Reproducibility Notes

- `chroma_db/` stores persisted vector index state.
- `logs/` can be used for run logs and diagnostics.
- For stricter reproducibility, pin exact versions in `requirements.txt` after you finalize your experiment environment.

## Help & Support

If you are extending experiments, start by checking:

- `.env` configuration
- input file locations under `data/`
- Neo4j database state (fresh ingestion before GraphRAG run)

For runtime failures, include:

- full command used
- stack trace
- current working directory
- relevant `.env` variable names (not secret values)

## Contributing

Contributions are welcome, especially around:

- path/config standardization across scripts
- packaging (`requirements.txt` / `pyproject.toml`)
- test coverage for evaluators and retrieval components
- CI checks and reproducible experiment automation

Recommended contribution flow:

1. Create a focused branch
2. Keep changes scoped to one phase/component
3. Add/update docs for behavior changes
4. Run affected scripts and attach outputs

## Maintainers

Project maintainer details should be finalized here.

- Primary maintainer: `Adil Usmani`
- Contact: `muhammadaadilusmani@gmail.com`

If this repository is being used for an academic thesis, replace this section with the official supervisor/student contact details.

