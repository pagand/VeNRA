import os
import re
import pickle
from typing import List, Optional
from llama_parse import LlamaParse
from venra.models import DocBlock, TextBlock, TableBlock, BlockType
from venra.logging_config import logger
from dotenv import load_dotenv

load_dotenv()

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

    async def parse_pdf(self, file_path: str) -> List[DocBlock]:
        """
        Parses a PDF and returns a list of DocBlocks with section hierarchy.
        """
        logger.info(f"Starting LlamaParse for: {file_path}")
        # LlamaParse.aload_data returns a list of Document objects
        documents = await self.parser.aload_data(file_path)
        
        all_blocks = []
        header_stack = []
        
        for doc in documents:
            content = doc.text
            # Simple line-by-line walker to track headers and content
            lines = content.split("\n")
            
            current_chunk = []
            
            for line in lines:
                header_match = re.match(r"^(#+)\s+(.*)", line)
                if header_match:
                    self._flush_chunk(current_chunk, header_stack, all_blocks)
                    current_chunk = []
                    
                    level = len(header_match.group(1))
                    title = header_match.group(2).strip()
                    
                    # Update header stack
                    header_stack = header_stack[:level-1]
                    header_stack.append(title)
                    logger.debug(f"Header Stack: {header_stack}")
                    continue

                # Table line detection: more inclusive to keep table together
                is_table_line = "|" in line
                
                if is_table_line:
                    # If we were in a non-table chunk, flush it
                    if current_chunk and not any("|" in l for l in current_chunk):
                        self._flush_chunk(current_chunk, header_stack, all_blocks)
                        current_chunk = []
                else:
                    # If we were in a table chunk, and this is a non-blank text line, flush it
                    if line.strip() and current_chunk and any("|" in l for l in current_chunk):
                        self._flush_chunk(current_chunk, header_stack, all_blocks)
                        current_chunk = []
                
                current_chunk.append(line)
            
            # Final flush for the document
            self._flush_chunk(current_chunk, header_stack, all_blocks)
            current_chunk = []
                    
        return all_blocks

    def _flush_chunk(self, lines: List[str], stack: List[str], all_blocks: List[DocBlock]):
        if not lines:
            return
        content = "\n".join(lines).strip()
        if not content:
            return
            
        # Determine if it's a table: contains | AND a separator line or multiple pipes
        has_pipe = any("|" in l for l in lines)
        has_separator = any("|" in l and "---" in l for l in lines)
        
        if has_pipe and has_separator:
            all_blocks.append(self._create_table_block(lines, stack))
        else:
            all_blocks.append(self._create_text_block(lines, stack))

    def _create_text_block(self, lines: List[str], stack: List[str]) -> TextBlock:
        return TextBlock(
            content="\n".join(lines).strip(),
            section_path=list(stack)
        )

    def _create_table_block(self, lines: List[str], stack: List[str]) -> TableBlock:
        return TableBlock(
            content="\n".join(lines).strip(),
            section_path=list(stack)
        )

    def save_dom(self, blocks: List[DocBlock], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(blocks, f)
        logger.info(f"DOM saved to {output_path}")

    @staticmethod
    def load_dom(input_path: str) -> List[DocBlock]:
        with open(input_path, "rb") as f:
            return pickle.load(f)
