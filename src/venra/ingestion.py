"""
venra/ingestion.py
PDF → DOM parser using LlamaParse.

Changes (Phase 2 — Semantic Chunking Upgrades):
  • _flush_chunk: Newline-preference boundary — never splits mid-sentence.
    If accumulated content exceeds MAX_CHUNK_CHARS, the split point is walked
    back to the most recent newline rather than cutting at an arbitrary index.
  • parse_pdf: Trailing-buffer injection — prepends the last
    TRAILING_BUFFER_CHARS characters of the previous TextBlock as
    [Previous Context: <buffer>] to bridge chunk boundaries and resolve
    dangling footnotes / coreference chains.
"""

import os
import re
import pickle
import hashlib
from typing import List, Optional

from llama_parse import LlamaParse

from venra.models import DocBlock, TextBlock, TableBlock, BlockType
from venra.logging_config import logger
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Maximum characters before we try to split a text chunk at a newline
MAX_CHUNK_CHARS: int = 3_000

# How many trailing characters of the previous block to prepend as context
TRAILING_BUFFER_CHARS: int = 300


class StructuralParser:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("LLAMA_CLOUD_API_KEY")
        if not self.api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY not found.")

        self.parser = LlamaParse(
            api_key=self.api_key,
            result_type="markdown",
            num_workers=4,
            verbose=True,
            language="en",
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def parse_pdf(self, file_path: str) -> List[DocBlock]:
        """
        Parses a PDF and returns a list of DocBlocks with section hierarchy.

        Phase-2 change: maintains a `trailing_buffer` string that is updated
        after every TextBlock flush and injected into the next TextBlock's
        content as [Previous Context: <buffer>].
        """
        logger.info(f"Starting LlamaParse for: {file_path}")
        documents = await self.parser.aload_data(file_path)

        all_blocks: List[DocBlock] = []
        header_stack: List[str] = []
        trailing_buffer: str = ""   # ← Phase-2: rolling last-300-char buffer

        for doc in documents:
            content = doc.text
            lines = content.split("\n")

            current_chunk: List[str] = []

            for line in lines:
                header_match = re.match(r"^(#+)\s+(.*)", line)
                if header_match:
                    # Flush before entering a new section
                    trailing_buffer = self._flush_chunk(
                        current_chunk,
                        header_stack,
                        all_blocks,
                        trailing_buffer=trailing_buffer,
                    )
                    current_chunk = []

                    level = len(header_match.group(1))
                    title = header_match.group(2).strip()
                    header_stack = header_stack[: level - 1]
                    header_stack.append(title)
                    logger.debug(f"Header Stack: {header_stack}")
                    continue

                is_table_line = "|" in line

                if is_table_line:
                    # Switching from text → table: flush accumulated text
                    if current_chunk and not any("|" in l for l in current_chunk):
                        trailing_buffer = self._flush_chunk(
                            current_chunk,
                            header_stack,
                            all_blocks,
                            trailing_buffer=trailing_buffer,
                        )
                        current_chunk = []
                else:
                    # Switching from table → text: flush accumulated table
                    if line.strip() and current_chunk and any(
                        "|" in l for l in current_chunk
                    ):
                        trailing_buffer = self._flush_chunk(
                            current_chunk,
                            header_stack,
                            all_blocks,
                            trailing_buffer=trailing_buffer,
                        )
                        current_chunk = []

                current_chunk.append(line)

            # Final flush for the document
            trailing_buffer = self._flush_chunk(
                current_chunk,
                header_stack,
                all_blocks,
                trailing_buffer=trailing_buffer,
            )
            current_chunk = []

        return all_blocks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_chunk(
        self,
        lines: List[str],
        stack: List[str],
        all_blocks: List[DocBlock],
        trailing_buffer: str = "",
    ) -> str:
        """
        Flush accumulated lines into one or more DocBlocks.

        Phase-2 changes:
        1. Newline-preference boundary: text chunks that exceed
           MAX_CHUNK_CHARS are split at the most recent newline within the
           limit rather than at an arbitrary character offset.
        2. Trailing-buffer injection: for TextBlocks only, the supplied
           `trailing_buffer` is prepended as [Previous Context: <buffer>].

        Returns:
            The new trailing buffer (last TRAILING_BUFFER_CHARS of the raw
            content just flushed), to be passed into the next call.
        """
        if not lines:
            return trailing_buffer

        content = "\n".join(lines).strip()
        if not content:
            return trailing_buffer

        # Determine block kind
        has_pipe = any("|" in l for l in lines)
        has_separator = any("|" in l and "---" in l for l in lines)
        is_table = has_pipe and has_separator

        if is_table:
            block = self._create_table_block(lines, stack)
            block.id = self._hash_block(block)
            all_blocks.append(block)
            # Fix 10: update the trailing buffer with the table's raw content.
            # Table headers carry critical context (scale factors, column years)
            # that the next text block needs in order to resolve references like
            # "as shown in the table above" or dangling footnotes.
            return content[-TRAILING_BUFFER_CHARS:]
        else:
            # ----------------------------------------------------------------
            # Phase-2 (1): Newline-preference chunking for long text blocks
            # ----------------------------------------------------------------
            segments = self._split_at_newlines(content, MAX_CHUNK_CHARS)

            current_buffer = trailing_buffer
            for i, segment in enumerate(segments):
                # Fix 11 (Phase-2 fix): every segment — not just the first —
                # gets a [Previous Context:] prefix from the immediately
                # preceding content.
                #   • Segment 0 gets the cross-block trailing buffer.
                #   • Segments 1, 2, … get the tail of the previous segment.
                # This ensures that long sections split into 3+ chunks each
                # have continuity, not just the first chunk.
                if current_buffer:
                    buffered_content = (
                        f"[Previous Context: {current_buffer}]\n\n{segment}"
                    )
                else:
                    buffered_content = segment

                block = TextBlock(
                    content=buffered_content,
                    section_path=list(stack),
                )
                block.id = self._hash_block(block)
                all_blocks.append(block)

                # The next segment's context is this segment's raw tail
                current_buffer = segment[-TRAILING_BUFFER_CHARS:]

            # Return the tail of the last raw segment as the new cross-block buffer
            return segments[-1][-TRAILING_BUFFER_CHARS:] if segments else trailing_buffer

    # ------------------------------------------------------------------
    # Newline-preference splitter
    # ------------------------------------------------------------------

    @staticmethod
    def _split_at_newlines(text: str, max_chars: int) -> List[str]:
        """
        Split `text` into segments of at most `max_chars` characters,
        always breaking at the most recent newline so sentences are never
        cut mid-way.

        If a single paragraph is longer than `max_chars` with no newline,
        it is kept intact (we prefer slightly oversized chunks over broken
        sentences).
        """
        if len(text) <= max_chars:
            return [text]

        segments: List[str] = []
        remaining = text

        while len(remaining) > max_chars:
            # Look for the last newline within the budget
            window = remaining[:max_chars]
            split_pos = window.rfind("\n")

            if split_pos == -1:
                # No newline in window — keep the whole paragraph together
                # and try to find the *next* newline after the budget
                next_nl = remaining.find("\n", max_chars)
                if next_nl == -1:
                    # No newline at all — emit the entire remainder as one chunk
                    segments.append(remaining.strip())
                    return segments
                split_pos = next_nl

            segment = remaining[:split_pos].strip()
            if segment:
                segments.append(segment)
            remaining = remaining[split_pos:].lstrip("\n")

        if remaining.strip():
            segments.append(remaining.strip())

        return segments

    # ------------------------------------------------------------------
    # Block factories
    # ------------------------------------------------------------------

    def _create_text_block(self, lines: List[str], stack: List[str]) -> TextBlock:
        return TextBlock(
            content="\n".join(lines).strip(),
            section_path=list(stack),
        )

    def _create_table_block(self, lines: List[str], stack: List[str]) -> TableBlock:
        return TableBlock(
            content="\n".join(lines).strip(),
            section_path=list(stack),
        )

    @staticmethod
    def _hash_block(block: DocBlock) -> str:
        id_seed = f"{block.section_path}_{block.content}"
        return hashlib.md5(id_seed.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_dom(self, blocks: List[DocBlock], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(blocks, f)
        logger.info(f"DOM saved to {output_path}")

    @staticmethod
    def load_dom(input_path: str) -> List[DocBlock]:
        with open(input_path, "rb") as f:
            return pickle.load(f)