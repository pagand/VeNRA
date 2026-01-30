# VeNRA: System Design & Architecture (MVP)

## 1. Executive Summary & Scientific Justification

**Problem:** LLMs suffer from "Stochastic Inaccuracy." In finance, a 99% accuracy rate results in 0% trust.
**Root Causes of Failure in Standard RAG:**
1.  **The Token Neighbor Problem:** LLMs swap "Million" and "Billion" because their vector representations are mathematically close.
2.  **Context Window Soup:** When retrieving multiple years of text, the LLM fails to bind specific numbers to specific dates ("Needle in a Haystack" failure).
3.  **Arithmetic Incompetence:** LLMs emulate the *language* of math, not the *logic* of math.

**Solution:** The **Verifiable Numerical Reasoning Agent (VeNRA)**.
We move from "Retrieving Text" to "Retrieving Variables." We replace probabilistic arithmetic with deterministic Python execution via a **Universal Fact Ledger (UFL)**.

**Objective:** Deliver an auditable, zero-hallucination QA system for 10-K filings using a "Filter, Project, Verify" methodology.

---

## 2. System Architecture



### Component A: The Hierarchical Ingestion Engine (Offline)

*Goal: Create a "Structure-Preserving" Knowledge Base.*

1.  **Tooling:** `LlamaParse` (Markdown mode) + `LlamaIndex` + Python.
2.  **The "Structure Preserver":**
    * We do not naively chunk text. We parse the PDF into a **Document Tree**.
    * **Context Stack:** We maintain a stack of headers (e.g., `['MD&A', 'Liquidity']`) during parsing. When a table is found, this stack is injected into the table's metadata.
3.  **The Outcome:**
    * **UFL (Parquet):** A deterministic table of variables (`row_id`, `value`, `period`, `metric`).
    * **Vector Store (ChromaDB):** Text chunks that contain *explicit links* (`contains_rows: [row_ids]`) back to the UFL.

### Component B: The "Smart Filter" Retrieval (Runtime)

*Goal: Deterministic scoping of data before the LLM sees it.*
*Constraint:* **DO NOT use Regex matching** (e.g., `df.contains('Debt')`). It fails on synonyms like "Obligations."

**Step 1: Entity Resolution (The Alias Map)**
* **Input:** User query "Apple."
* **Logic:** Look up "Apple" in the **Page 1 Metadata Map** (generated during ingestion).
* **Result:** Resolve to Canonical ID `ID_AAPL`.

**Step 2: Semantic Schema Search (The Metric Selector)**
* **Input:** User query "Debt."
* **Logic:** Perform a **Vector Search** against the *unique metric names* present in the UFL columns.
* **Result:** The vector search returns `['Senior Notes', 'Revolving Credit Facilities', 'Total Term Debt']`.
* **Action:** Filter the UFL DataFrame to include only these specific metrics.

**Step 3: Linked Context Retrieval**
* **Action:** Retrieve the text chunks associated with the filtered UFL rows using `source_chunk_id`.
**Graph Expansion (Runtime)**

* **Logic:** Use the Fact Ledger as a graph (in Pandas).

* *Algorithm:* `df[df['entity_primary'].isin(initial_entities) & df['relationship'] == 'supplier_of']`

* **Why:** If the user asks about "Apple's risk," we simply filter the DataFrame to find connected entities (Suppliers, Subsidiaries) and add *their* risk chunks to the context. **No GraphDB required.**

### Component C: The Program-Aided Agent (The "Cortex")

*Goal: Context-Aware Code Execution.*

* **Environment:** `E2B` Sandbox or Local Python Subprocess.
* **Input:**
    1.  The Filtered UFL (presented as a clean Markdown table).
    2.  The Retrieved Text (Source of Truth).
    3.  The Nuance Notes (extracted from table footnotes).
* **System Prompt:**
    > "You are a financial analyst.
    > 1. **Check Nuance:** Read the 'Nuance Notes' column. If a value is 'Unaudited' or 'Restated', note it.
    > 2. **Cross-Check:** Compare the Table Value against the Text Chunk. If the text says 'excluding one-time tax', but the table includes it, adjust your logic.
    > 3. **Execute:** Write Python code using the variables provided. Handle missing data by raising an error. **Do not guess numbers.**"

### Component D: The SLM Judge (Hallucination Detection)

*Goal: Low-latency, specialized verification.*

* **Model:** Small Language Model (e.g., Qwen-4B, Phi-3), potentially fine-tuned.
* **Input:**
    1.  **User Query:** "What is the 2023 Revenue?"
    2.  **UFL Context:** The rows used for calculation.
    3.  **Retrieved Text:** The raw source chunks.
    4.  **Generated Answer:** "2023 Revenue is $383B."
* **Mechanism:**
    *   The SLM acts as a binary classifier or regressor.
    *   It outputs a **Groundedness Score** (0.0 to 1.0).
    *   *Why SLM?* Faster and cheaper than calling GPT-4 for every verification step. Can be trained on specific "financial consistency" tasks.
* **Output:**
    *   Score > Threshold (e.g., 0.9): Pass.
    *   Score < Threshold: Trigger "I am not sure" or fallback to manual review.

---

## 3. Evaluation Methodology

1.  **Hallucination Rate (Numerical):** % of answers where the calculated number deviates from Ground Truth.
2.  **Retrieval Precision:** Did the "Semantic Schema Search" find the right column? (e.g., finding "Obligations" when asked for "Debt").
3.  **Judge Accuracy:** How well does the SLM Proxy correlate with human labels of hallucination?

---

## 4. Critical Implementation Risks

1.  **The "Metric Gap":** If Vector Search fails to map "Debt" to "Notes Payable", the UFL returns an empty DataFrame.
    *   *Mitigation:* Use a specialized embedding model for finance (e.g., `bge-m3` or `uae-large-v1`) for the schema search.
2.  **Context Misalignment:** If LlamaParse misinterprets a table header, the `period` column will be wrong.
    *   *Mitigation:* The "Hierarchy Stack" logic in ingestion is the primary defense.
3.  **SLM Overconfidence:** The small model might just agree with the generated answer.
    *   *Mitigation:* Use "Chain of Verification" prompts or fine-tune on a dataset of *known* hallucinations.


### **Research Questions (RQs)**



1. **RQ1 (Data Structure):** Does a "Fact Ledger" (semi-structured intermediate layer) outperform pure Unstructured RAG on multi-hop financial queries?

2. **RQ2 (Hybrid Retrieval):** Can runtime "Graph Expansion" (via Pandas filtering) recover context lost by vector similarity search?

3. **RQ3 (Reliability):** Does Program-Aided Generation (Code execution) reduce numerical hallucination rates to near zero compared to Chain-of-Thought (CoT)?