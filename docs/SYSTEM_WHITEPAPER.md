# VeNRA System Whitepaper: Experimental Architecture & Data

This document provides a technical overview of the VeNRA (Verifiable Numerical Reasoning Agent) system, documenting the core components used in research experiments, their operational logic, and the datasets supporting the evaluation.

---

## 1. System Components (`src/venra`)

The following modules form the core RAG and Reasoning pipeline evaluated in **Stage A** and **Stage B**.

### A. Entry Points & APIs
| Component | Logic | Functionality / Invocation |
| :--- | :--- | :--- |
| **`pipeline.py`** | `IngestionPipeline` orchestration. | Run via: `python -m venra.pipeline <pdf_path>`. Executes PDF $\rightarrow$ DOM $\rightarrow$ UFL $\rightarrow$ ChromaDB. |
| **`main.py`** | FastAPI server (`SentinelJudge`). | Run via: `uvicorn venra.main:app`. Exposes `POST /verify` for checking answers against context and traces. |

### B. Retrieval Layer
| Component | Logic | Input | Output |
| :--- | :--- | :--- | :--- |
| **`Navigator`** | SLM-based planning. Translates intent into structured clues. | User Query (str) | `RetrievalPlan` (JSON) |
| **`DualRetriever`** | Parallel search in Vector DB & UFL with **Lexical Gating** (recall > 0.30). | `RetrievalPlan` | `List[UFLRow]`, `List[DocBlock]` |
| **`ContextAssembler`** | Deduplication, ranking (keyword + UFL linkage), and markdown formatting. | Raw retrieval results | Augmented Context (Markdown) |

### C. Execution Layer
| Component | Logic | Input | Output |
| :--- | :--- | :--- | :--- |
| **`ReasoningAgent`** | Two-pass Flow: Kimi NIM (Code) + Groq (Synthesis). Trust Text over UFL if conflicting. | Query + Context | `FinalResponse` + Code Trace |
| **`PythonExecutor`** | Local sandbox for deterministic math via `exec()`. | Python Code (str) | Result (float/str) or Error |

### D. Ingestion Engine (The Knowledge Base)
| Component | Logic | Functionality |
| :--- | :--- | :--- |
| **`StructuralParser`** | LlamaParse + Stack Machine | PDF $\rightarrow$ Hierarchical DOM (Blocks) |
| **`TableMelter`** | Deterministic Pandas melting with scale/unit normalization. | TableBlock $\rightarrow$ `UFLRow` (EXACT) |
| **`TextSynthesizer`** | SLM + **Double-Lock Aligner** (Mechanical + Semantic). | TextBlock $\rightarrow$ `UFLRow` (Verified) |

---

## 2. Resilient SLM Orchestration

To support high-throughput benchmarking and minimize transient API failures (Rate Limits, 504 Timeouts), VeNRA implements a **High-Resiliency SLM Layer**.

### A. Multi-Key Pooling
The system supports simultaneous use of multiple API keys across all providers:
*   **Groq Pool:** Automatically cycles through `GROQ_API_KEY`, `GROQ_API_KEY_2...7` to bypass per-key daily/minute token limits.
*   **NVIDIA Pool:** Cycles through `NVIDIA_API_KEY` and `NVIDIA_API_KEY_2` for reliable Kimi NIM access.
*   **Centralized Config:** All key pools are managed centrally in `src/venra/config.py`.

### B. Aggressive Retry & Backoff
All SLM-dependent components (`Navigator`, `ReasoningAgent`, `ResilientTextSynthesizer`) use the **Tenacity** retry framework with:
*   **Exponential Backoff:** Random jitter between 1s and 10s.
*   **Deterministic Failure:** Silent "Safe Plan" fallbacks have been removed. The system now "Fails Loudly" or Retries until success to ensure benchmark purity.
*   **Deep Reasoning Timeouts:** The `ReasoningAgent` (using Kimi k2.5 via NVIDIA NIM) utilizes an extended 300-second timeout to accommodate long chain-of-thought execution. It captures the raw `reasoning` trace for auditability and limits retries to 3 attempts to fail faster on genuine API outages.

---

## 3. Data Schemas & Distribution

### A. Core Internal Schemas (Pydantic)
The system relies on strict typing to guarantee determinism between components.

#### 1. The Universal Fact Ledger Row (`UFLRow` v2.0)
This is the API contract between the Ingestion Engine and the Reasoning Agent.
| Field | Type | Description |
| :--- | :--- | :--- |
| `row_id` | str | Deterministic hash for deduplication. |
| `canonical_entity_id` | str | Normalized ID (e.g. 'ID_AAPL') for graph traversal. |
| `metric_name` | str | Semantic key (e.g. 'Revenue'). |
| `num_value` | float \| None | Pure float. None if fact is qualitative. |
| `grounding_quote` | str | Verbatim substring from text (Lock 1). |
| `unit_normalized` | str | USD, USD/Share, Ratio, Percent, etc. |
| `scale` | float | Multiplier (e.g. 1e6 for millions). |
| `period_start` | str \| None | ISO-8601 start date. |
| `period_end` | str \| None | ISO-8601 end date. |
| `text_nuance` | str \| None | Footnotes, restatements, or conditions. |
| `source_chunk_id` | str | Foreign key to ChromaDB chunk. |
| `alignment_status` | str | EXACT, PARTIAL, FUZZY, UNALIGNED. |
| `confidence_score` | float | Reliability (0.95 for tables, 0.70 for text). |

#### 2. The Retrieval Plan (`RetrievalPlan`)
Output of the Navigator SLM, instructing the DualRetriever.
| Field | Type | Description |
| :--- | :--- | :--- |
| `ufl_query` | UFLFilter | Structured parameters: `entity_ids`, `metric_keywords`, `years`, `nuance_focus`. |
| `vector_hypothesis` | str | "Hallucinated" text chunk used for dense vector search. |
| `vector_keywords` | List[str] | Boost keywords for BM25/hybrid filtering. |

### B. Experimental Dataset Schema (JSONL)
All experimental datasets are normalized to this schema to ensure a single execution loop during evaluation.

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | str | Unique record identifier. |
| `query` | str | The user question. |
| `context_chunks` | List[str] | The source text/tables (Golden Evidence). |
| `trace_code` | str | (Optional) Python math trace for logic verification. |
| `target_sentence` | str | The golden answer scalar or span. |
| `metadata` | Dict | Dataset-specific tags (company, year, doc_type). |

### B. Experimental Dataset Distribution
| Dataset | Count | Role in Experiments |
| :--- | :--- | :--- |
| **FinanceBench** | 1,150 | Semantic Conflation (Text-only RAG). |
| **FinQA** | 8,281 | Pure Mathematical Logic & Code Tracing. |
| **TAT-QA** | 16,474 | Hybrid Tabular/Textual Reasoning. |
| **Phantom** | 1,982 | Natural Hallucination baseline. |
| **TruthfulQA** | 114 | Axiomatic Knowledge / Self-Awareness. |
| **Total** | **28,001** | **Unified Benchmark Pool** |

---

## 3. Operational Logic (The 2x2 Matrix)

For **Experiment 1 & 2**, the system is executed in four specific configurations to isolate variables:

1.  **Baseline RAG:** ChromaDB (Vector) $\rightarrow$ LLM (Chain-of-Thought).
2.  **Smart Retrieval, Dumb Math:** DualRetriever (VeNRA) $\rightarrow$ LLM (Chain-of-Thought).
3.  **Dumb Retrieval, Smart Math:** ChromaDB (Vector) $\rightarrow$ VeNRA Code Agent.
4.  **VeNRA Full:** DualRetriever (VeNRA) $\rightarrow$ VeNRA Code Agent.

---

## 4. Metric Definitions

### Retrieval Metrics (Exp 2)
*   **Golden Recall@K:** Boolean check if the retrieved context contains the gold standard evidence.
*   **Semantic Bleed Ratio:** Ratio of irrelevant metrics with high semantic similarity to query.

### Generation Metrics (Exp 1)
*   **Exact Match (EM):** Numerical regex match against `target_sentence`.
*   **Failure Type 1:** Retrieval Failure (Blindness).
*   **Failure Type 2:** Generative Conflation (Distracted by adjacent noise).
*   **Failure Type 3:** Arithmetic Hallucination (Logic error).
