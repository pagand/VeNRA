"""
experiments/build_global_index.py
Phase 1: Build Global Confusion Index
--------------------------------------
Ingests ALL context_chunks from FinanceBench, TAT-QA, and FinQA into a
single ChromaDB collection and UFL parquet.  Measures Double-Lock Aligner
purity as Ablation 1 for the paper.  Emits schema_summary.json consumed
by the Navigator in Phase 2.

Directory contract:
  experiments/build_global_index.py          ← this file
  experiments/resilient_synthesizer.py       ← imported here
  src/venra/                                 ← library, import only
  data/golden_records/                       ← read-only normalized JSONL
  data/exp/global_index/chroma_db/
  data/exp/global_index/ufl.parquet
  data/exp/global_index/ufl_checkpoint.jsonl
  data/exp/global_index/chunk_metadata.json
  data/exp/global_index/schema_summary.json  ← metrics_by_record format
  data/exp/results/ingestion_purity.json

IMPORT NOTE: both TableMelter and ContextIndexer live in venra.synthesis.
Do not move this import line.

ALL FIXES (accumulated across all sessions):
  1.  ContextIndexer import confirmed as venra.synthesis.
  2.  Checkpoint stores block_type to keep text-only purity counters clean.
  3.  ChromaDB indexing calls index_blocks(blocks_to_process), not full list.
  4.  is_table replaced with is_table_aggressive() from resilient_synthesizer.
  5.  extract_facts_chunked() used for blocks > MAX_CHUNK_CHARS.
  6.  DEBUG_LIMIT is a strict hard cap; trimmed after collection.
  7.  Document-level sampling: DEBUG_LIMIT selects WHOLE DOCUMENTS.
      Prior chunk-level sampling produced partial documents that Phase 2's
      100%-indexed filter rejected entirely, leaving zero eligible queries.
  8.  Metadata written AFTER processing, excluding failed blocks.
  9.  Failed blocks tracked via success flag from process_text_block.
      Best-effort ChromaDB delete attempted; metadata exclusion protects
      Phase 2 eligibility regardless of whether delete succeeds.
  10. purity_summary["total_ufl_rows"] uses len(df) (post-dedup parquet).
  11. ENTITY ID FIX (CRITICAL): canonical_entity_id patched to
      "ID_<COMPANY_UPPER>" per block after extraction. Without this fix all
      UFL rows carry "EXP_GLOBAL"; the Navigator generates company-specific
      IDs so every UFL query returns zero results, making Run 2 ≡ Run 1 and
      Run 4 ≡ Run 3 — the entire 2×2 matrix is scientifically invalid.
  12. extract_facts_chunked returns 3-tuple (rows, count, any_failed) matching
      the updated extract_facts_with_proposals signature. process_text_block
      does NOT checkpoint a partial result when any_failed=True.
  13. Schema summary uses metrics_by_record (per-document metric list).
      The Navigator reads this to build targeted RetrievalPlan.ufl_query
      parameters. A flat global metric list was useless — "Net Income" from
      AAPL and MSFT are indistinguishable without the record ID anchor.
  14. NUMERIC METRIC FILTER: TableMelter rows where metric_name is a bare
      number (e.g. "166.0") are dropped after melt(). These arise when a
      table's first column contains numbers rather than labels (malformed
      header). If kept, they appear in the VeNRA context as
      `| row_id | 166.0 | chunk_id |` with no semantic label, giving the
      agent no actionable signal. They also pollute schema_summary with
      numeric keys the Navigator cannot use.
  15. Sub-chunk suffix stripping in schema_summary: split_large_block gives
      sub-blocks IDs like `{parent_id}_f820`. source_chunk_id in UFLRows
      reflects the sub-block ID; active_meta is keyed by parent chunk IDs.
      Strip the 4-hex suffix before metadata lookup so metrics_by_record
      is populated correctly.
  16. CANONICAL WHITESPACE NORMALIZATION: get_chunk_id now collapses all
      whitespace sequences and lowercases before hashing. This fixes the
      FinQA eligibility anomaly (0.1% rate) caused by formatting mismatches.
"""

import collections
import hashlib
import json
import os
import re
import random
import asyncio
import sys
from typing import Any, Dict, List, Set, Tuple

from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from venra.models import DocBlock, BlockType, UFLRow
from venra.synthesis import TableMelter, ContextIndexer   # DO NOT change this import
from venra.logging_config import logger
from venra.schema import is_numeric_metric

from resilient_synthesizer import ResilientTextSynthesizer, is_table_aggressive

# ── Paths ─────────────────────────────────────────────────────────────────────
GLOBAL_IDX_DIR  = "data/exp/global_index"
RESULTS_DIR     = "data/exp/results"
CHROMA_DB_PATH  = os.path.join(GLOBAL_IDX_DIR, "chroma_db")
UFL_PATH        = os.path.join(GLOBAL_IDX_DIR, "ufl.parquet")
CHECKPOINT_PATH = os.path.join(GLOBAL_IDX_DIR, "ufl_checkpoint.jsonl")
METADATA_PATH   = os.path.join(GLOBAL_IDX_DIR, "chunk_metadata.json")
SCHEMA_PATH     = os.path.join(GLOBAL_IDX_DIR, "schema_summary.json")
PURITY_PATH     = os.path.join(RESULTS_DIR, "ingestion_purity.json")

GOLDEN_RECORDS_DIR = "data/golden_records"
DATASETS = [
    "financebench_normalized.jsonl",
    "tatqa_normalized_test_gold.jsonl",
    "finqa_normalized.jsonl",
]

CONCURRENCY_LIMIT = 2    # 4 keys × ~1.2 req/s ≈ 5 safe concurrent
DEBUG_LIMIT       = 900  # Targeting 200-query yield (300 per dataset)
MAX_CHUNKS_PER_DOC = 30  # Cap on massive documents to ensure distractor diversity
MAX_CHUNK_CHARS   = 1500 # Blocks above this are split before SLM extraction.
RANDOM_SEED       = 42


# ── Entity ID helpers (FIX 11) ────────────────────────────────────────────────

def _company_to_entity_id(company: str) -> str:
    """
    Convert a display company name to the canonical ID format generated by
    the Navigator SLM.

    "Activision Blizzard"  → "ID_ACTIVISION_BLIZZARD"
    "AES Corporation"      → "ID_AES_CORPORATION"
    "Global_Entity"        → "EXP_GLOBAL"   (fallback for records with no company)

    The DualRetriever filters the UFL by entity_ids from RetrievalPlan.
    If all rows carry "EXP_GLOBAL" the filter never matches and Run 2 / Run 4
    retrieve identical context to Run 1 / Run 3 — the 2×2 comparison is invalid.
    """
    if not company or company in ("Global_Entity", ""):
        return "EXP_GLOBAL"
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", company)
    clean = re.sub(r"\s+", "_", clean.strip()).upper()
    return f"ID_{clean}"


def _patch_entity_ids(rows: List[UFLRow], block: DocBlock, chunk_meta: Dict[str, Any]) -> List[UFLRow]:
    """
    Overwrite canonical_entity_id, entity_name_raw, and source_record_id 
    on every row. source_record_id is critical for retrieval scoping 
    when entity resolution fails (Bug 2).
    """
    if not rows:
        return rows
    
    # We now fetch company cleanly from chunk_meta, which was populated perfectly at load time
    meta = chunk_meta.get(block.id, {})
    company = meta.get("company", "Global_Entity")
    entity_id = _company_to_entity_id(company)
    
    # Extract source records from metadata
    record_ids = meta.get("source_records", [])
    
    for row in rows:
        row.canonical_entity_id = entity_id
        row.entity_name_raw     = company
        row.company_label       = entity_id # Lock the CANONICAL ID at write time
        # We store the first record ID as a primary anchor; 
        # Multi-record chunks are rare in this evaluation.
        if record_ids:
            # We inject a custom field that will be saved to parquet
            setattr(row, "source_record_id", record_ids[0])
            
    return rows


def _is_numeric_metric_name(name: str) -> bool:
    """
    Returns True if metric_name is a bare number — a TableMelter artifact from
    tables whose first column contains values rather than labels.
    These rows have no semantic label so the Navigator cannot use them and the
    agent cannot interpret them.  Drop them before writing to the UFL parquet.
    """
    try:
        float(name)
        return True
    except (ValueError, TypeError):
        return False


# ── Chunk ID (single source of truth) ────────────────────────────────────────

def get_chunk_id(record_id: str, content: str) -> str:
    """
    Deterministic MD5 with Canonical Whitespace Normalization and Namespacing. 
    Includes record_id in the hash to prevent cross-entity mega-chunk fusion 
    (where identical boilerplate from 13,000 companies collapses into one chunk, 
    destroying metadata and inflating recall).
    """
    if not content:
        return ""
    # Canonicalise: lowercase, strip, and collapse internal whitespace
    canonical = " ".join(content.lower().split())
    # Namespace with record_id
    namespaced = f"{record_id}::{canonical}"
    return hashlib.md5(namespaced.encode()).hexdigest()


# ── Checkpoint I/O ────────────────────────────────────────────────────────────

def save_checkpoint(
    chunk_id: str,
    block_type: str,
    accepted_rows: List[UFLRow],
    proposed_count: int,
) -> None:
    with open(CHECKPOINT_PATH, "a") as f:
        f.write(json.dumps({
            "chunk_id":       chunk_id,
            "block_type":     block_type,
            "proposed_count": proposed_count,
            "accepted_rows":  [r.model_dump() for r in accepted_rows],
        }) + "\n")


def load_checkpoint() -> Tuple[Set[str], List[UFLRow], int, int]:
    """
    Returns (processed_ids, all_ufl_rows, total_proposed_text, total_accepted_text).
    TABLE blocks excluded from text counters (deterministic, no rejection rate).
    """
    processed_ids, all_ufl_rows = set(), []
    total_proposed_text = total_accepted_text = 0

    if not os.path.exists(CHECKPOINT_PATH):
        return processed_ids, all_ufl_rows, total_proposed_text, total_accepted_text

    logger.info(f"Loading checkpoint from {CHECKPOINT_PATH}…")
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data       = json.loads(line)
                cid        = data["chunk_id"]
                btype      = data.get("block_type", "TEXT")
                proposed   = data["proposed_count"]
                chunk_rows = [UFLRow(**r) for r in data["accepted_rows"]]
                processed_ids.add(cid)
                all_ufl_rows.extend(chunk_rows)
                if btype == "TEXT":
                    total_proposed_text += proposed
                    total_accepted_text += len(chunk_rows)
            except Exception as e:
                logger.error(f"Checkpoint parse error (skipping): {e}")

    logger.info(
        f"Checkpoint: {len(processed_ids)} chunks, "
        f"{total_proposed_text} text proposed, {total_accepted_text} accepted."
    )
    return processed_ids, all_ufl_rows, total_proposed_text, total_accepted_text


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_chunks() -> Tuple[
    List[DocBlock],
    Dict[str, Dict[str, Any]],
    Dict[str, List[str]],
    Dict[str, str],
]:
    """
    Returns (all_blocks, chunk_meta, record_to_chunks, record_to_ds).
    record_to_chunks enables document-level sampling in DEBUG_LIMIT logic.
    """
    all_blocks:       List[DocBlock]            = []
    chunk_meta:       Dict[str, Dict[str, Any]] = {}
    record_to_chunks: Dict[str, List[str]]      = {}
    record_to_ds:     Dict[str, str]            = {}
    seen_hashes:      Set[str]                  = set()

    for ds_name in DATASETS:
        path = os.path.join(GOLDEN_RECORDS_DIR, ds_name)
        if not os.path.exists(path):
            logger.warning(f"Dataset not found: {path} — skipping.")
            continue

        logger.info(f"Loading {ds_name}…")
        ds_tag = ds_name.split("_")[0]

        with open(path) as f:
            for line in f:
                record  = json.loads(line)
                
                # FIX: Inbound filter for FinanceBench 'Unfounded' records.
                # These are misclassified model failures, not real unanswerable queries.
                if ds_name == "financebench_normalized.jsonl" and record.get("label") == "Unfounded":
                    continue

                rec_id  = record["id"]
                
                # FIX: Entity resolution (Issue 4 & 11)
                # FinQA IDs are formatted like "finqa_TICKER/YYYY/page_X.pdf-Y"
                # FinanceBench has metadata.company
                metadata = record.get("metadata", {})
                company  = metadata.get("company")
                if not company or company == "Global_Entity":
                    # Try doc_name fallback (FinanceBench)
                    doc_name = metadata.get("doc_name", "")
                    if doc_name:
                        company = doc_name.split("_")[0]
                
                if not company or company == "Global_Entity":
                    # Try FinQA ticker regex
                    m = re.match(r"finqa_([A-Z0-9]+)/", rec_id)
                    if m:
                        company = m.group(1)
                
                if not company:
                    company = "Global_Entity"

                rec_chunk_ids: List[str] = []

                for chunk_text in record["context_chunks"]:
                    cid = get_chunk_id(rec_id, chunk_text)
                    rec_chunk_ids.append(cid)
                    if cid not in seen_hashes:
                        block_type = (
                            BlockType.TABLE if is_table_aggressive(chunk_text)
                            else BlockType.TEXT
                        )
                        all_blocks.append(DocBlock(
                            id=cid,
                            content=chunk_text,
                            block_type=block_type,
                            section_path=["Experiment", ds_tag, company],
                            page_num=0,
                        ))
                        seen_hashes.add(cid)
                    if cid not in chunk_meta:
                        chunk_meta[cid] = {"source_records": [], "company": company}
                    chunk_meta[cid]["source_records"].append(rec_id)

                record_to_chunks[rec_id] = list(dict.fromkeys(rec_chunk_ids))
                record_to_ds[rec_id]     = ds_tag

    logger.info(f"Total unique chunks: {len(all_blocks)}")
    return all_blocks, chunk_meta, record_to_chunks, record_to_ds


# ── Block processing ──────────────────────────────────────────────────────────

async def process_text_block(
    block: DocBlock,
    text_synth: ResilientTextSynthesizer,
    semaphore: asyncio.Semaphore,
    chunk_meta: Dict[str, Any],
) -> Tuple[str, List[UFLRow], int, bool]:
    """
    Extracts facts from a text block, patches entity IDs, checkpoints.

    Returns (block_id, accepted_rows, proposed_count, success).
    success=False means all retries exhausted — do NOT checkpoint, force retry
    next run.  success=True with rows=[] is valid (no extractable facts).
    """
    async with semaphore:
        # Rate limit protection for freemium keys
        await asyncio.sleep(0.5)
        
        context_str = f"Registrant: {block.section_path[-1]}."
        try:
            accepted, proposed_count, any_failed = await text_synth.extract_facts_chunked(
                block,
                context_str=context_str,
                max_chars=MAX_CHUNK_CHARS,
            )
            if any_failed:
                logger.error(
                    f"Block {block.id[:8]}… had sub-chunk failures — "
                    "discarding partial result, will retry next run."
                )
                return block.id, [], 0, False

            accepted = _patch_entity_ids(accepted, block, chunk_meta)
            save_checkpoint(block.id, "TEXT", accepted, proposed_count)
            return block.id, accepted, proposed_count, True

        except Exception as e:
            logger.error(f"Unhandled failure on block {block.id[:8]}…: {e}")
            return block.id, [], 0, False


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    logger.info("Phase 1: Building Global Confusion Index…")
    os.makedirs(GLOBAL_IDX_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR,    exist_ok=True)

    # 1. Load all blocks
    blocks, chunk_meta, record_to_chunks, record_to_ds = load_all_chunks()

    # 2. Resume
    processed_ids, all_ufl_rows, total_proposed_text, total_accepted_text = load_checkpoint()
    logger.info(f"Resuming: {len(processed_ids)} / {len(blocks)} already done.")

    # 3. Document-level stratified sampling (FIX 7)
    # Selects WHOLE DOCUMENTS until per-dataset budget is reached.
    # A document is skipped entirely if its chunks don't fit within the budget.
    # This guarantees Phase 2's 100%-indexed eligibility filter has records.
    if DEBUG_LIMIT is not None:
        logger.info(
            f"DEBUG: targeting {DEBUG_LIMIT} chunks via document-level sampling "
            f"(seed={RANDOM_SEED})."
        )
        rng = random.Random(RANDOM_SEED)
        block_by_id: Dict[str, DocBlock] = {b.id: b for b in blocks}

        ds_records: Dict[str, List[str]] = collections.defaultdict(list)
        for rec_id, ds_tag in record_to_ds.items():
            ds_records[ds_tag].append(rec_id)

        n_datasets    = max(1, len(ds_records))
        target_per_ds = DEBUG_LIMIT // n_datasets
        remainder     = DEBUG_LIMIT % n_datasets
        sampled_blocks: List[DocBlock] = []

        for i, (ds_tag, rec_ids) in enumerate(ds_records.items()):
            chunk_budget = target_per_ds + (1 if i < remainder else 0)

            fully_processed, needs_processing = [], []
            for rec_id in rec_ids:
                cids = record_to_chunks.get(rec_id, [])
                if not cids or len(cids) > MAX_CHUNKS_PER_DOC:
                    continue
                if all(cid in processed_ids for cid in cids):
                    fully_processed.append((rec_id, cids))
                else:
                    needs_processing.append((rec_id, cids))

            rng.shuffle(needs_processing)
            selected_chunk_ids: Set[str]      = set()
            selected_rec_ids:   List[str]     = []
            ds_blocks:          List[DocBlock] = []

            # Pass 1: already-done records first (zero marginal cost)
            for rec_id, cids in fully_processed:
                if len(selected_chunk_ids) + len(cids) <= chunk_budget:
                    for cid in cids:
                        if cid in block_by_id and cid not in selected_chunk_ids:
                            ds_blocks.append(block_by_id[cid])
                            selected_chunk_ids.add(cid)
                    selected_rec_ids.append(rec_id)

            # Pass 2: new complete documents until budget reached
            for rec_id, cids in needs_processing:
                if len(selected_chunk_ids) >= chunk_budget:
                    break
                if len(selected_chunk_ids) + len(cids) <= chunk_budget:
                    for cid in cids:
                        if cid in block_by_id and cid not in selected_chunk_ids:
                            ds_blocks.append(block_by_id[cid])
                            selected_chunk_ids.add(cid)
                    selected_rec_ids.append(rec_id)

            sampled_blocks.extend(ds_blocks)
            logger.info(
                f"  {ds_tag}: {len(selected_chunk_ids)} chunks from "
                f"{len(selected_rec_ids)} complete docs (budget={chunk_budget})."
            )

        if len(sampled_blocks) > DEBUG_LIMIT:
            logger.warning(f"Trimming {len(sampled_blocks)} → {DEBUG_LIMIT} (hard cap).")
            sampled_blocks = sampled_blocks[:DEBUG_LIMIT]

        blocks = sampled_blocks
        logger.info(f"Working set: {len(blocks)} blocks.")

    if not blocks:
        logger.warning("No blocks to process.")
        return

    # 4. Split new vs done
    blocks_to_process = [b for b in blocks if b.id not in processed_ids]
    logger.info(
        f"  Already processed: {len(blocks) - len(blocks_to_process)}\n"
        f"  Remaining        : {len(blocks_to_process)}"
    )

    failed_block_ids: List[str] = []

    if not blocks_to_process:
        logger.info("All blocks already processed — skipping synthesis.")
    else:
        # 5. Vector indexing (new blocks only)
        logger.info(f"Indexing {len(blocks_to_process)} new chunks into ChromaDB…")
        indexer = ContextIndexer(db_path=CHROMA_DB_PATH)
        
        # Build a record map for indexing: cid -> rec_id
        # This enables strict document-level scoping in retrieve()
        record_map = {}
        for rec_id, cids in record_to_chunks.items():
            for cid in cids:
                record_map[cid] = rec_id
                
        indexer.index_blocks(blocks_to_process, record_map=record_map)

        # 6. UFL synthesis
        # Both TableMelter and ResilientTextSynthesizer use entity_id="EXP_GLOBAL"
        # (shared instances). Per-block entity IDs are patched via _patch_entity_ids
        # after extraction — constructing per-block instances would be wasteful.
        melter     = TableMelter(entity_id="EXP_GLOBAL", entity_name_raw="Global Experiment Entity")
        text_synth = ResilientTextSynthesizer(entity_id="EXP_GLOBAL", entity_name_raw="Global Experiment Entity")
        semaphore  = asyncio.Semaphore(CONCURRENCY_LIMIT)

        table_new         = 0
        text_blocks_order: List[DocBlock] = []
        text_tasks:        List           = []

        # Progress bar 1: Routing (Table vs Text)
        routing_pbar = tqdm(blocks_to_process, desc="Routing chunks", unit="chunk")
        for block in routing_pbar:
            if block.block_type == BlockType.TABLE:
                rows = melter.melt(block)

                # FIX 14: drop rows with numeric metric_names (malformed table
                # headers produce rows like metric_name="166.0" that have no
                # semantic label and cannot be used by the Navigator or agent).
                rows = [r for r in rows if not _is_numeric_metric_name(str(r.metric_name))]

                rows = _patch_entity_ids(rows, block, chunk_meta)   # FIX 11 + Record Scoping
                all_ufl_rows.extend(rows)
                save_checkpoint(block.id, "TABLE", rows, len(rows))
                table_new += 1
            else:
                text_blocks_order.append(block)
                text_tasks.append(
                    asyncio.create_task(
                        process_text_block(block, text_synth, semaphore, chunk_meta)
                    )
                )
        routing_pbar.close()

        if text_tasks:
            # Progress bar 2: Live SLM extraction
            text_pbar = tqdm(
                total=len(text_tasks),
                desc="Extracting (TEXT)",
                unit="block",
            )
            
            # Helper to update counts in real-time
            ok_count   = 0
            fail_count = 0
            rows_added = 0

            # Using as_completed for live updates
            for future in asyncio.as_completed(text_tasks):
                block_id, accepted_rows, proposed_count, success = await future
                
                if success:
                    all_ufl_rows.extend(accepted_rows)
                    total_proposed_text += proposed_count
                    total_accepted_text += len(accepted_rows)
                    ok_count += 1
                    rows_added += len(accepted_rows)
                else:
                    fail_count += 1
                    failed_block_ids.append(block_id)
                
                text_pbar.update(1)
                text_pbar.set_postfix({
                    "ok": ok_count, 
                    "fail": fail_count, 
                    "rows": rows_added
                })
            
            text_pbar.close()

        logger.info(
            f"Processed — table: {table_new}, text: {len(text_tasks)}, "
            f"failures: {len(failed_block_ids)}."
        )

        # 7. ChromaDB / UFL sync for failed blocks (FIX 9)
        if failed_block_ids:
            logger.warning(
                f"\n{'='*60}\n"
                f"SYNC WARNING: {len(failed_block_ids)} block(s) in ChromaDB "
                f"with NO UFL entries (all retries exhausted).\n"
                f"IDs: {failed_block_ids[:10]}"
                f"{'  …more' if len(failed_block_ids) > 10 else ''}\n"
                f"Excluded from chunk_metadata.json — Phase 2 eligibility protected.\n"
                f"{'='*60}"
            )
            try:
                indexer.delete_blocks(failed_block_ids)
                logger.info(f"ChromaDB: removed {len(failed_block_ids)} orphaned blocks.")
            except (AttributeError, NotImplementedError):
                logger.warning(
                    "ContextIndexer has no delete_blocks(). "
                    "Orphaned vectors remain; metadata exclusion still protects Phase 2."
                )
            except Exception as e:
                logger.error(f"ChromaDB cleanup error: {e}. Metadata exclusion still applies.")

    # 8. Write chunk_metadata AFTER processing (FIX 8)
    # BUG 7 FIX: Ensure both sub-chunk IDs and parent IDs exist in metadata.
    # This ensures Phase 2 eligibility check (which uses parent IDs) succeeds.
    failed_ids_set = set(failed_block_ids)
    current_ids    = {b.id for b in blocks} - failed_ids_set
    
    # Map both parent and sub-chunk IDs to source records
    active_meta = {}
    for cid, m in chunk_meta.items():
        if cid in current_ids:
            active_meta[cid] = m
            # If this was a parent, sub-chunks will refer back to it in Phase 2
    
    with open(METADATA_PATH, "w") as f:
        json.dump(active_meta, f, indent=2)
    logger.info(
        f"Metadata: {len(active_meta)} chunks "
        f"({len(failed_ids_set)} failed excluded)."
    )

    # 9. UFL parquet
    if all_ufl_rows:
        df = pd.DataFrame([r.model_dump() for r in all_ufl_rows])
        # BUG 6 FIX: Deduplicate by content key, not by row_id.
        # Identical facts extracted multiple times (e.g. from sub-chunks or near-duplicates)
        # must collapse to a single row to save tokens and avoid agent confusion.
        df = df.drop_duplicates(subset=["canonical_entity_id", "metric_name", "period_end", "num_value"])
        df.to_parquet(UFL_PATH, index=False)
        logger.info(f"UFL: {len(df)} rows → {UFL_PATH}")
    else:
        df = pd.DataFrame()
        logger.warning("No UFL rows produced.")

    # 10. Schema summary — metrics_by_record (FIX 13 + FIX 15)
    # Navigator reads metrics_by_record[rec_id] to build targeted UFL queries.
    if not df.empty:
        metrics_by_record: Dict[str, List[str]] = {}
        periods_by_record: Dict[str, List[str]] = {}
        global_entities:   Set[str]             = set()
        global_periods:    Set[str]             = set()
        metric_counts: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        period_sets:   Dict[str, Set[str]]            = collections.defaultdict(set)

        for _, row in df.iterrows():
            m_name = str(row["metric_name"])
            p_end  = str(row["period_end"]) if pd.notna(row["period_end"]) else None

            if _is_numeric_metric_name(m_name) or is_numeric_metric(m_name):
                continue

            cid      = str(row["source_chunk_id"])
            base_cid = re.sub(r"_[0-9a-f]{4}$", "", cid)
            meta     = active_meta.get(base_cid) or active_meta.get(cid)
            if not meta:
                continue

            global_entities.add(str(row["canonical_entity_id"]))
            if p_end:
                global_periods.add(p_end)

            for rec_id in meta.get("source_records", []):
                metric_counts[rec_id][m_name] += 1
                if p_end:
                    period_sets[rec_id].add(p_end)

        for rec_id, counts in metric_counts.items():
            sorted_m = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            metrics_by_record[rec_id] = [m for m, _ in sorted_m[:500]]
            periods_by_record[rec_id] = sorted(list(period_sets[rec_id]))

        schema_summary = {
            "total_ufl_rows":    len(df),
            "entity_ids":        sorted(global_entities),
            "metrics_by_record": metrics_by_record,
            "periods_by_record": periods_by_record,
            "period_ends":       sorted(global_periods),
            "alignment_counts":  df["alignment_status"].value_counts().to_dict(),
        }
    else:
        schema_summary = {
            "total_ufl_rows": 0, "entity_ids": [],
            "metrics_by_record": {}, "period_ends": [], "alignment_counts": {},
        }

    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema_summary, f, indent=2)
    logger.info(f"Schema summary → {SCHEMA_PATH}")

    # 11. Ingestion purity report (FIX 10)
    alignment_dist = (
        pd.Series([r.alignment_status for r in all_ufl_rows]).value_counts().to_dict()
        if all_ufl_rows else {}
    )
    rejection_rate = (
        (total_proposed_text - total_accepted_text) / total_proposed_text
        if total_proposed_text > 0 else 0.0
    )
    total_processed = len(processed_ids) + len(blocks_to_process) - len(failed_block_ids)

    purity_summary = {
        "total_chunks_in_run":        len(blocks),
        "total_chunks_processed":     total_processed,
        "total_chunks_failed":        len(failed_block_ids),
        "failed_chunk_ids":           failed_block_ids,
        "total_ufl_rows":             len(df),          # post-dedup parquet count (FIX 10)
        "total_text_facts_proposed":  total_proposed_text,
        "total_text_facts_accepted":  total_accepted_text,
        "double_lock_rejection_rate": round(rejection_rate, 4),
        "alignment_distribution":     alignment_dist,
    }
    with open(PURITY_PATH, "w") as f:
        json.dump(purity_summary, f, indent=2)

    logger.info(
        f"Phase 1 complete.\n"
        f"  Working set      : {len(blocks)} blocks\n"
        f"  Processed        : {total_processed}\n"
        f"  Failed           : {len(failed_block_ids)}\n"
        f"  Text proposed    : {total_proposed_text}\n"
        f"  Text accepted    : {total_accepted_text}\n"
        f"  Rejection rate   : {rejection_rate:.1%}\n"
        f"  UFL rows (dedup) : {len(df)}\n"
        f"  Purity report    → {PURITY_PATH}"
    )


if __name__ == "__main__":
    asyncio.run(main())
