import os
import asyncio
from typing import List, Optional
import pandas as pd
from venra.ingestion import StructuralParser
from venra.synthesis import EntityResolver, TableMelter, TextSynthesizer, ContextIndexer
from venra.schema import SchemaGenerator
from venra.models import BlockType, DocBlock, UFLRow
from venra.logging_config import logger
from venra.config import settings

class IngestionPipeline:
    def __init__(self):
        self.parser = StructuralParser()
        self.resolver = EntityResolver()
        self.indexer = ContextIndexer()
        self.schema_gen = SchemaGenerator(output_path=os.path.join(settings.DATA_DIR, "processed/schema_summary.json"))

    async def run(self, pdf_path: str, skip_parsing: bool = False):
        """
        Runs the full ingestion pipeline: PDF -> DOM -> UFL -> Vector DB -> Schema Summary.
        """
        base_name = os.path.basename(pdf_path).replace(".pdf", "")
        dom_path = os.path.join(settings.DATA_DIR, "processed", f"{base_name}_dom.pkl")
        ufl_path = os.path.join(settings.DATA_DIR, "processed", f"{base_name}_ufl.parquet")
        schema_path = os.path.join(settings.DATA_DIR, "processed", f"{base_name}_schema_summary.json")
        
        # Configure Schema Gen for this specific file
        self.schema_gen.output_path = schema_path

        # 0. Check if UFL already exists and we want to skip
        if skip_parsing and os.path.exists(ufl_path):
            logger.info(f"UFL already exists at {ufl_path}. Skipping extraction.")
            df = pd.read_parquet(ufl_path)
            all_ufl_rows = [UFLRow(**r) for r in df.to_dict('records')]
            
            # Ensure schema summary is generated if missing or for consistency
            if not os.path.exists(schema_path):
                logger.info(f"Regenerating missing schema summary from existing UFL...")
                self.schema_gen.add_rows(all_ufl_rows)
                self.schema_gen.save()
            return all_ufl_rows

        # 1. Structural Parsing
        if skip_parsing and os.path.exists(dom_path):
            logger.info(f"Loading existing DOM from {dom_path}")
            blocks = StructuralParser.load_dom(dom_path)
        else:
            logger.info(f"Parsing PDF: {pdf_path}")
            blocks = await self.parser.parse_pdf(pdf_path)
            self.parser.save_dom(blocks, dom_path)

        # 2. Entity Resolution
        entity_meta = await self.resolver.resolve_entity(blocks)
        self.schema_gen.add_entity(entity_meta)
        
        # Determine Current Year from cover page context or metadata
        # Simple heuristic: look for 4-digit numbers in the first 20 blocks
        current_year = "UNKNOWN"
        for b in blocks[:20]:
            match = re.search(r"202[0-9]", b.content)
            if match:
                current_year = match.group(0)
                break
        
        context_str = f"Registrant: {entity_meta.official_name}. Current Fiscal Year: {current_year}. Dollars in millions unless specified."
        logger.info(f"Using Global Context: {context_str}")

        # 3. Knowledge Synthesis (UFL & Vector Indexing)
        melter = TableMelter(entity_id=entity_meta.canonical_id, entity_name_raw=entity_meta.official_name)
        text_synth = TextSynthesizer(entity_id=entity_meta.canonical_id, entity_name_raw=entity_meta.official_name)
        
        all_ufl_rows = []

        # Index text blocks in ChromaDB
        self.indexer.index_blocks(blocks)

        for block in blocks:
            if block.block_type == BlockType.TABLE:
                logger.info(f"Processing table in {block.section_path}")
                table_rows = melter.melt(block)
                all_ufl_rows.extend(table_rows)
            elif block.block_type == BlockType.TEXT:
                # OPTIMIZATION: Only extract facts from text that looks like it has financial substance
                content = block.content
                has_money = "$" in content
                digits = sum(c.isdigit() for c in content)
                
                if has_money or digits > 4:
                    logger.info(f"Extracting facts from text in {block.section_path[:2]}...")
                    text_facts = await text_synth.extract_facts(block, context_str=context_str)
                    all_ufl_rows.extend(text_facts)
                    # Small sleep to avoid hitting Groq TPM limits too hard
                    await asyncio.sleep(0.5)

        # 4. Save UFL to Parquet
        if all_ufl_rows:
            df = pd.DataFrame([r.dict() for r in all_ufl_rows])
            df.to_parquet(ufl_path, index=False)
            logger.info(f"UFL saved to {ufl_path} ({len(df)} rows)")
        
        # 5. Schema Summary Generation
        self.schema_gen.add_rows(all_ufl_rows)
        self.schema_gen.save()
        
        logger.info("Pipeline execution complete.")
        return all_ufl_rows

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m venra.pipeline <pdf_path>")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    pipeline = IngestionPipeline()
    asyncio.run(pipeline.run(pdf_path))
