"""
venra/retriever.py
Dual retrieval from UFL (structured) and ChromaDB (vector) with:
  • Relational expansion (entity pivoting, frequency-based chunk expansion)
  • [Phase 5] Lexical Pre-Filtering — Counter token-intersection gate on all
    vector matches and UFL metric matches to prevent semantically-close but
    lexically-wrong pairings (e.g. "Net Sales" → "Net Income").

Design decisions (captured in UFL_spec §3 addendum):
  • LEXICAL_OVERLAP_THRESHOLD (default 0.30): minimum recall of query tokens
    that must appear in the candidate text/metric before the match is kept.
    Recall (not Jaccard) is used because query terms are fewer than document
    tokens; Jaccard is dominated by document-only tokens and collapses to near-
    zero even for correct matches.

  • Stop-words — "net" is a stop-word (intentional and correct):
    Both "Net Income" and "Net Sales" contain "net", making it non-discriminative.
    Filtering it forces the gate to operate on the discriminative tokens
    ("income" vs "sales") where it actually belongs.

    Proof why "net" must be filtered:
      If "net" were NOT a stop-word, a "Net Income" query would tokenise to
      ["net", "income"]. A "Net Sales" candidate would contain ["net", "sales"].
      Recall = 1/2 = 0.50 → above threshold → INCORRECTLY ACCEPTED.
      With "net" as a stop-word: query=["income"], candidate=["sales"],
      recall=0.0 → CORRECTLY REJECTED.

  • [Previous Context:] prefix stripping: the trailing-buffer prefix injected
    by Phase 2 ingestion is removed from block.content before tokenization.
    Including prefix tokens could cause false acceptances: a chunk irrelevant to
    the query might pass because its injected prefix (from an adjacent,
    query-relevant block) shares tokens with the query.

  • nuance_focus (UFLFilter field): when set, applied as a case-insensitive
    substring filter on the text_nuance column. This lets the Navigator narrow
    UFL results to e.g. "Restated" or "Adjusted" rows. NaN text_nuance rows
    are excluded when nuance_focus is active.
"""

from __future__ import annotations

import collections
import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import chromadb

from venra.models import RetrievalPlan, UFLRow, DocBlock, BlockType
from venra.config import settings
from venra.logging_config import logger

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

LEXICAL_OVERLAP_THRESHOLD: float = 0.20

# Strips the [Previous Context: ...]\n\n prefix injected by Phase 2 ingestion.
_PREV_CONTEXT_RE = re.compile(r"^\[Previous Context:[^\]]*\]\s*", re.DOTALL)

# Stop-words carrying no discriminative signal.
# Financial-specific inclusions explained in module docstring above.
_STOP_WORDS: frozenset = frozenset(
    {
        # Standard English
        "the", "a", "an", "of", "in", "and", "or", "for", "to", "by",
        "on", "at", "from", "with", "as", "is", "are", "was", "were",
        "its", "our",
        # Financial non-discriminative: present in too many unrelated metrics.
        # "net"   → in "Net Income", "Net Sales", "Net Revenue" — see module docstring.
        # "total" → in "Total Assets", "Total Revenue", etc.
        # "per"   → connector in "Earnings Per Share".
        "net", "total", "per",
    }
)


# ---------------------------------------------------------------------------
# Module-level helpers (importable for direct unit testing)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """
    Fast regex word tokenizer (lower-cased, stop-word filtered).
    Uses a word-boundary regex — no BPE/SentencePiece overhead.
    Tokens shorter than 2 characters are discarded to avoid single-letter noise.
    """
    raw = re.findall(r"\b[a-zA-Z0-9&%/]+\b", text.lower())
    return [t for t in raw if t not in _STOP_WORDS and len(t) > 1]


def _strip_prev_context_prefix(text: str) -> str:
    """
    Remove the [Previous Context: ...] prefix so only the chunk's own content
    participates in the lexical gate. See module docstring for the full rationale.
    """
    return _PREV_CONTEXT_RE.sub("", text)


def _lexical_recall(query_tokens: List[str], candidate_text: str) -> float:
    """
    recall = |Counter intersection| / |query_tokens|

    candidate_text is automatically stripped of its [Previous Context:] prefix
    before tokenization.

    Returns 1.0 when query_tokens is empty (vacuously true — do not filter).
    """
    if not query_tokens:
        return 1.0
    clean_text = _strip_prev_context_prefix(candidate_text)
    query_counter = collections.Counter(query_tokens)
    cand_counter = collections.Counter(_tokenize(clean_text))
    intersection = sum(min(query_counter[t], cand_counter[t]) for t in query_counter)
    return intersection / sum(query_counter.values())


def _passes_lexical_gate(
    query_tokens: List[str],
    candidate_text: str,
    threshold: float = LEXICAL_OVERLAP_THRESHOLD,
) -> bool:
    """Return True if the candidate achieves ≥ threshold lexical recall."""
    score = _lexical_recall(query_tokens, candidate_text)
    logger.debug(
        f"Lexical recall={score:.2f} threshold={threshold} "
        f"candidate='{candidate_text[:60]}'"
    )
    return score >= threshold


# ---------------------------------------------------------------------------
# DualRetriever
# ---------------------------------------------------------------------------

class DualRetriever:
    """
    Parallel retrieval from UFL (structured) and ChromaDB (vector) with
    Relational Expansion and Lexical Pre-Filtering.

    Phase 5 additions:
    ─────────────────
    • _apply_lexical_filter(): static method wrapping any list of DocBlocks.
      Drops candidates whose content (prefix-stripped) does not achieve
      LEXICAL_OVERLAP_THRESHOLD recall against query tokens.
    • All four retrieval arms gated: hypothesis chunks, keyword-boost chunks,
      entity pivot chunks, UFL fuzzy metric matches.
    • Expansion B (frequency-based) and C (completeness) bypass the gate —
      these rows/chunks are already grounded by primary retrieval.
    • nuance_focus from UFLFilter is now applied in _query_ufl.
    """

    def __init__(
        self,
        file_prefix: Optional[str] = None,
        ufl_path: Optional[str] = None,
        db_path: Optional[str] = None,
    ):
        if ufl_path:
            self.ufl_path = os.path.abspath(ufl_path)
        elif file_prefix:
            self.ufl_path = os.path.join(
                settings.DATA_DIR, "processed", f"{file_prefix}_ufl.parquet"
            )
        else:
            self.ufl_path = os.path.join(settings.DATA_DIR, "processed/ufl.parquet")

        self.db_path = db_path or settings.CHROMA_DB_PATH

        if os.path.exists(self.ufl_path):
            self.df = pd.read_parquet(self.ufl_path)
            logger.info(f"Retriever loaded UFL with {len(self.df)} rows.")
        else:
            self.df = pd.DataFrame()
            logger.warning(f"UFL file not found at {self.ufl_path}.")

        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.text_collection = self.chroma_client.get_or_create_collection(
            "venra_text_chunks"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        plan: RetrievalPlan,
        k: int = 4,
        include_all_chunks_for_ufl: bool = True,
        include_all_ufl_for_chunks: bool = True,
        doc_id: Optional[str] = None,
        company: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Dual retrieval with Relational Expansion and Lexical Pre-Filtering.
        """
        logger.info(
            f"Starting retrieval for query: {plan.vector_hypothesis[:50]}... (k={k})"
        )

        hypothesis_tokens = _tokenize(plan.vector_hypothesis)
        keyword_tokens = _tokenize(" ".join(plan.vector_keywords or []))
        combined_query_tokens = list(
            (
                collections.Counter(hypothesis_tokens)
                + collections.Counter(keyword_tokens)
            ).keys()
        )

        # 1. Core similarity
        raw_hypothesis_chunks = self._query_vector(plan.vector_hypothesis, k=k, doc_id=doc_id)
        hypothesis_chunks = self._apply_lexical_filter(
            raw_hypothesis_chunks, combined_query_tokens, label="hypothesis"
        )

        # 1b. Keyword boost
        effective_k_keywords = max(k, 5)
        keyword_chunks: List[DocBlock] = []
        if plan.vector_keywords:
            keyword_query = " ".join(plan.vector_keywords)
            logger.info(f"Keyword Boost Search: '{keyword_query}' (k={effective_k_keywords})")
            raw_keyword_chunks = self._query_vector(keyword_query, k=effective_k_keywords, doc_id=doc_id)
            keyword_chunks = self._apply_lexical_filter(
                raw_keyword_chunks, combined_query_tokens, label="keyword-boost"
            )

        chunk_id_map: Dict[str, DocBlock] = {c.id: c for c in hypothesis_chunks}
        for c in keyword_chunks:
            chunk_id_map.setdefault(c.id, c)

        # BUG 3 FIX: Toggle first_pass_miss if vector search failed to find ANY chunks
        first_pass_miss = (len(chunk_id_map) == 0)

        # 2. Direct UFL query
        ufl_query = plan.ufl_query
        selected_ufl_rows = (
            self._query_ufl(
                ufl_query, 
                combined_query_tokens, 
                doc_id_scope=doc_id,
                company_scope=company
            )
            if ufl_query
            else []
        )
        
        # BUG 2 FIX: Cap UFL results to prevent context inflation (Production Rule §4)
        if len(selected_ufl_rows) > 10:
            logger.warning(f"UFL result set too large ({len(selected_ufl_rows)}). Capping to 10.")
            selected_ufl_rows = selected_ufl_rows[:10]

        row_id_map: Dict[str, UFLRow] = {r.row_id: r for r in selected_ufl_rows}

        # 3. Expansion A — related entity pivoting
        related_entities = list(
            {r.related_entity_id for r in selected_ufl_rows if r.related_entity_id}
        )
        for entity in related_entities:
            raw_entity_chunks = self._query_vector(f"Information about {entity}", k=2)
            entity_tokens = _tokenize(entity)
            filtered = self._apply_lexical_filter(
                raw_entity_chunks,
                entity_tokens if entity_tokens else combined_query_tokens,
                label=f"entity-pivot:{entity}",
                threshold=0.20,
            )
            for ec in filtered:
                chunk_id_map.setdefault(ec.id, ec)

        # 3. Expansion B — UFL → Chunk (frequency-based, bypasses gate)
        if include_all_chunks_for_ufl and selected_ufl_rows:
            source_counts = collections.Counter(
                r.source_chunk_id for r in selected_ufl_rows
            )
            new_candidate_ids = [
                cid for cid, _ in source_counts.most_common() if cid not in chunk_id_map
            ]
            for cid in new_candidate_ids[:3]:
                expanded = self._fetch_chunks_by_ids([cid])
                if expanded:
                    chunk_id_map[cid] = expanded[0]

        # 3. Expansion C — Chunk → UFL (completeness, bypasses gate)
        # BUG 4 FIX: Scope expansion to doc_id to avoid multi-record contamination.
        if include_all_ufl_for_chunks and chunk_id_map:
            current_chunk_ids = list(chunk_id_map.keys())
            if not self.df.empty and "source_chunk_id" in self.df.columns:
                # FIX: Handle sub-chunk suffixes (e.g. {parent_id}_f820)
                # We strip the suffix from the UFL column before matching against ChromaDB parent IDs.
                ufl_source_ids = self.df["source_chunk_id"].astype(str)
                base_source_ids = ufl_source_ids.str.replace(r"_[0-9a-f]{4}$", "", regex=True)
                expanded_rows = self.df[base_source_ids.isin(current_chunk_ids)]
                
                if doc_id:
                    expanded_rows = expanded_rows[expanded_rows["source_record_id"] == doc_id]
                
                for _, er in expanded_rows.iterrows():
                    row_obj = UFLRow(**er.to_dict())
                    row_id_map.setdefault(row_obj.row_id, row_obj)

        final_rows = list(row_id_map.values())
        final_chunks = list(chunk_id_map.values())

        # BUG 2 & BUG 4 FIX: Detect UFL Bleed (Type 0 failure) on FINAL results
        ufl_bleed = False
        if final_rows:
            found_entities = {r.canonical_entity_id for r in final_rows}
            if plan.ufl_query and plan.ufl_query.entity_ids:
                if any(eid not in plan.ufl_query.entity_ids for eid in found_entities):
                    ufl_bleed = True
            elif len(found_entities) > 1:
                ufl_bleed = True
                
            # BUG 4: Scoping precision validation
            if doc_id:
                if any(r.source_record_id != doc_id for r in final_rows if r.source_record_id):
                    ufl_bleed = True
        
        if ufl_bleed:
             logger.error("UFL Bleed detected: results contain multiple or unrelated entities/records.")

        logger.info(
            f"Retrieval complete: {len(final_rows)} UFL rows, "
            f"{len(final_chunks)} text chunks."
        )

        return {
            "ufl_rows": final_rows,
            "text_chunks": final_chunks,
            "meta": {
                "ufl_count": len(final_rows),
                "text_count": len(final_chunks),
                "vector_keywords": plan.vector_keywords,
                "first_pass_miss": first_pass_miss,
                "ufl_bleed": ufl_bleed,
            },
        }

    # ------------------------------------------------------------------
    # Phase 5: Lexical Pre-Filter
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_lexical_filter(
        candidates: List[DocBlock],
        query_tokens: List[str],
        label: str = "",
        threshold: float = LEXICAL_OVERLAP_THRESHOLD,
    ) -> List[DocBlock]:
        """
        Drop DocBlocks whose content (after [Previous Context:] prefix stripping)
        does not achieve ≥ threshold lexical recall against query_tokens.

        If query_tokens is empty, all candidates pass (vacuously true).
        """
        if not query_tokens:
            return candidates

        kept, dropped = [], []
        for block in candidates:
            if _passes_lexical_gate(query_tokens, block.content, threshold):
                kept.append(block)
            else:
                dropped.append(block.id)

        if dropped:
            logger.info(
                f"[LexicalFilter/{label}] Dropped {len(dropped)} chunk(s) "
                f"below threshold={threshold:.2f}: {dropped}"
            )
        logger.info(
            f"[LexicalFilter/{label}] Kept {len(kept)}/{len(candidates)} chunk(s)."
        )
        return kept

    # ------------------------------------------------------------------
    # UFL query
    # ------------------------------------------------------------------

    def _query_ufl(
        self,
        filter_spec: UFLFilter,
        query_tokens: Optional[List[str]] = None,
        doc_id_scope: Optional[str] = None,
        company_scope: Optional[str] = None,
    ) -> List[UFLRow]:
        """
        Query the UFL DataFrame applying entity, year, metric, and nuance_focus
        filters in sequence.
        """
        if self.df.empty:
            return []

        mask = pd.Series(True, index=self.df.index)

        # 0. Permissive Company Filtering (Bug 2 T0 Bleed Fix)
        # Every fact must belong to the target registrant OR be a global table.
        if company_scope and "company_label" in self.df.columns:
            mask &= (
                (self.df["company_label"] == company_scope) | 
                (self.df["company_label"] == "EXP_GLOBAL")
            )

        # Ensure we never return explicitly hallucinated rows that failed grounding
        if "alignment_status" in self.df.columns:
            mask &= self.df["alignment_status"] != "UNALIGNED"

        if filter_spec.entity_ids and doc_id_scope and "source_record_id" in self.df.columns:
            # FIX: Robust fallback for ticker mismatches (e.g. ID_VZ vs ID_VERIZON)
            # If we have a doc_id_scope, we accept rows that match the entity OR the record.
            # This ensures that even if the Navigator picks a slightly different entity ID,
            # the primary record's data is still retrieved.
            entity_mask = self.df["canonical_entity_id"].isin(filter_spec.entity_ids)
            record_mask = self.df["source_record_id"] == doc_id_scope
            mask &= (entity_mask | record_mask)
        elif filter_spec.entity_ids:
            mask &= self.df["canonical_entity_id"].isin(filter_spec.entity_ids)
        elif doc_id_scope and "source_record_id" in self.df.columns:
            # BUG 2 & 4: True record-scoping fallback logic
            mask &= self.df["source_record_id"] == doc_id_scope

        if filter_spec.years:
            year_pattern = "|".join(filter_spec.years)
            period_col = (
                "period_end" if "period_end" in self.df.columns
                else "period_start" if "period_start" in self.df.columns
                else None
            )
            if period_col:
                mask &= self.df[period_col].astype(str).str.contains(
                    year_pattern, na=False
                )
        # BUG 5 FIX: treat empty years as all years (already true because mask not updated)

        if filter_spec.metric_keywords:
            exact_mask = self.df["metric_name"].isin(filter_spec.metric_keywords)

            if not exact_mask.any():
                pattern = "|".join(
                    f".*{re.escape(m)}.*" for m in filter_spec.metric_keywords
                )
                fuzzy_mask = self.df["metric_name"].str.contains(
                    pattern, case=False, na=False, regex=True
                )

                # Gate fuzzy matches by tokens of the metric keywords themselves,
                # NOT by the full query tokens (which may be very long/noisy).
                metric_tokens = _tokenize(" ".join(filter_spec.metric_keywords))
                if metric_tokens and fuzzy_mask.any():
                    fuzzy_rows = self.df[fuzzy_mask]
                    gate_mask = fuzzy_rows["metric_name"].apply(
                        lambda mn: _passes_lexical_gate(metric_tokens, mn)
                    )
                    passed_idx = fuzzy_rows[gate_mask].index
                    dropped_count = fuzzy_mask.sum() - len(passed_idx)
                    if dropped_count:
                        logger.info(
                            f"[LexicalFilter/ufl-metric] Dropped {dropped_count} "
                            f"fuzzy metric match(es) below threshold."
                        )
                    fuzzy_mask = self.df.index.isin(passed_idx)

                metric_mask = exact_mask | fuzzy_mask
            else:
                metric_mask = exact_mask

            mask &= metric_mask

        # nuance_focus: narrow to rows whose text_nuance contains the keyword.
        if getattr(filter_spec, "nuance_focus", None) and "text_nuance" in self.df.columns:
            nuance_mask = self.df["text_nuance"].str.contains(
                re.escape(filter_spec.nuance_focus),
                case=False,
                na=False,   # NaN text_nuance → False (excluded from nuanced results)
            )
            before = mask.sum()
            mask &= nuance_mask
            logger.info(
                f"[UFLFilter/nuance_focus] '{filter_spec.nuance_focus}': "
                f"{mask.sum()}/{before} row(s) retained."
            )

        results = self.df[mask]
        return [UFLRow(**r.to_dict()) for _, r in results.iterrows()]

    # ------------------------------------------------------------------
    # Vector store helpers
    # ------------------------------------------------------------------

    def _query_vector(self, hypothesis: str, k: int = 3, doc_id: Optional[str] = None) -> List[DocBlock]:
        # We deliberately DO NOT filter by doc_id here.
        # The vector search must remain GLOBAL to prove "Semantic Fusion"
        # vulnerabilities in standard RAG. Filtering by doc_id would artificially
        # boost the baseline's precision by preventing it from searching other companies.
        results = self.text_collection.query(
            query_texts=[hypothesis], 
            n_results=k
        )
        blocks: List[DocBlock] = []
        if not results["ids"] or not results["ids"][0]:
            return blocks
        for i in range(len(results["ids"][0])):
            blocks.append(
                DocBlock(
                    id=results["ids"][0][i],
                    content=results["documents"][0][i],
                    block_type=BlockType(results["metadatas"][0][i]["block_type"]),
                    section_path=json.loads(results["metadatas"][0][i]["section_path"]),
                    page_num=results["metadatas"][0][i].get("page_num"),
                )
            )
        return blocks

    def _fetch_chunks_by_ids(self, ids: List[str]) -> List[DocBlock]:
        if not ids:
            return []
        results = self.text_collection.get(ids=ids)
        blocks: List[DocBlock] = []
        for i in range(len(results["ids"])):
            blocks.append(
                DocBlock(
                    id=results["ids"][i],
                    content=results["documents"][i],
                    block_type=BlockType(results["metadatas"][i]["block_type"]),
                    section_path=json.loads(results["metadatas"][i]["section_path"]),
                    page_num=results["metadatas"][i].get("page_num"),
                )
            )
        return blocks