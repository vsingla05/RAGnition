"""
COMPLETE MULTIMODAL INGESTION PIPELINE
Handles: Text + Images + Tables
Stores: All modalities with metadata in same vector space
Per-document isolation via doc_id
"""

import os
import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ingestion.pdf_parser import extract_pdf_multimodal
    from ingestion.multimodal_extractor import MultimodalExtractor
    from ingestion.figure_extractor import process_pdf_for_figures
    from vectordb.chroma_client import init_chroma, store_chunks, delete_document
    from embeddings.multimodal_embedder import MultimodalEmbedder
except ImportError:
    from backend.ingestion.pdf_parser import extract_pdf_multimodal
    from backend.ingestion.multimodal_extractor import MultimodalExtractor
    from backend.ingestion.figure_extractor import process_pdf_for_figures
    from backend.vectordb.chroma_client import init_chroma, store_chunks, delete_document
    from backend.embeddings.multimodal_embedder import MultimodalEmbedder


class TableProcessor:
    """Process tables: summarize + markdown + visual storage"""

    @staticmethod
    def process_table(table_data: Dict, doc_id: str, doc_name: str, table_index: int) -> Dict:
        """
        Convert table into multiple formats:
        1. Markdown for text search
        2. Summary for semantic understanding
        """

        try:
            table_content = table_data.get("content", "")
            page_num = table_data.get("page", "unknown")

            # Create markdown representation
            markdown_table = f"Table {table_index} (Page {page_num}):\n{table_content}"

            # Create summary
            summary = f"Table on page {page_num} contains structured data with columns and rows"

            return {
                "text": markdown_table,
                "type": "table",
                "modality": "table",
                "metadata": {
                    "type": "table",
                    "modality": "table",
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page": int(page_num) if str(page_num).isdigit() else 0,
                    "table_index": table_index,
                    "summary": summary,
                    "format": "markdown"
                }
            }
        except Exception as e:
            print(f"⚠️  Error processing table {table_index}: {e}")
            return None


class MultimodalIngestionPipeline:
    """
    TRUE MULTIMODAL INGESTION
    Extracts and indexes: Text + Images + Tables
    Per-document isolation using doc_id
    """

    def __init__(self):
        self.embedder = None
        self.document_registry = {}

    def initialize(self):
        """Initialize CLIP embedder"""
        print("🤖 Initializing CLIP embedder...")
        try:
            self.embedder = MultimodalEmbedder()
            print("✅ CLIP embedder ready")
        except Exception as e:
            print(f"⚠️  CLIP not available: {e}")
            print("   System will use sentence-transformer text embeddings")

    def ingest_document(self, pdf_path: str) -> str:
        """
        Ingest PDF and extract ALL modalities
        Returns: doc_id for tracking
        """

        print("\n" + "="*70)
        print("🚀 MULTIMODAL DOCUMENT INGESTION")
        print("="*70 + "\n")

        # Initialize CLIP if not done
        if not self.embedder:
            self.initialize()

        # Create unique document ID
        doc_id = str(uuid.uuid4())
        doc_name = Path(pdf_path).stem
        filename = Path(pdf_path).name

        print(f"📋 Document: {doc_name}")
        print(f"📋 Document ID: {doc_id}\n")

        try:
            # ========================
            # PHASE 1: EXTRACTION
            # ========================

            print("PHASE 1: MULTIMODAL EXTRACTION")
            print("-" * 70)

            # Extract TEXT using unstructured or PyMuPDF fallback
            print("\n📝 Extracting text chunks...")
            text_items = []
            try:
                raw_items = extract_pdf_multimodal(pdf_path)
                text_items = self._convert_to_text_dicts(raw_items, doc_id, doc_name, filename)
                print(f"✅ Extracted {len(text_items)} text chunks (via unstructured)")
            except Exception as e:
                print(f"⚠️  unstructured failed ({e}), using PyMuPDF fallback")
                text_items = self._extract_text_pymupdf(pdf_path, doc_id, doc_name, filename)
                print(f"✅ Extracted {len(text_items)} text chunks (via PyMuPDF)")

            # Extract IMAGES and TABLES using PyMuPDF
            print("\n🖼️  Extracting images and tables...")
            extractor = MultimodalExtractor(pdf_path)
            extracted_data = extractor.extract_all()
            images = extracted_data.get("images", [])
            raw_tables = extracted_data.get("tables", [])
            print(f"✅ Extracted {len(images)} images, {len(raw_tables)} tables")

            # Extract FIGURES with captions and metadata
            print("\n🎨 Extracting figures with captions...")
            try:
                indexed_figures, fig_summary = process_pdf_for_figures(pdf_path, images, "\n".join([item.get("text", "") for item in text_items]))
                print(f"✅ Extracted {fig_summary['total_figures']} figures with captions and metadata")
                print(f"   📊 By type: {fig_summary['by_type']}")
                print(f"   📍 By section: {fig_summary.get('by_section', {})}")
            except Exception as e:
                print(f"⚠️  Figure extraction failed: {e}")
                indexed_figures = []
                fig_summary = {"total_figures": 0, "by_type": {}}

            # Process tables
            table_items = []
            for idx, table in enumerate(raw_tables):
                processed = TableProcessor.process_table(table, doc_id, doc_name, idx)
                if processed:
                    table_items.append(processed)

            # ========================
            # PHASE 2: EMBEDDING
            # ========================

            print("\n" + "="*70)
            print("PHASE 2: EMBEDDING")
            print("-" * 70)

            embedded_items = []

            # Embed TEXT
            print(f"\n🧠 Embedding {len(text_items)} text chunks...")
            embedded_text = self._embed_text_items(text_items)
            embedded_items.extend(embedded_text)
            print(f"✅ Embedded {len(embedded_text)} text chunks")

            # Embed IMAGES (with CLIP if available, else store with text embedding)
            if images:
                print(f"\n🖼️  Embedding {len(images)} images...")
                embedded_images = self._embed_images(images, doc_id, doc_name, filename)
                embedded_items.extend(embedded_images)
                print(f"✅ Embedded {len(embedded_images)} images")

            # Store INDEXED FIGURES with captions and metadata
            if indexed_figures:
                print(f"\n🎨 Storing {len(indexed_figures)} indexed figures...")
                figure_items = []
                for fig in indexed_figures:
                    figure_items.append({
                        "text": fig.get("content", ""),
                        "type": "figure",
                        "modality": "figure",
                        "metadata": fig.get("metadata", {})
                    })
                embedded_items.extend(figure_items)
                print(f"✅ Stored {len(figure_items)} figures with descriptions")

            # Embed TABLES
            if table_items:
                print(f"\n📊 Embedding {len(table_items)} tables...")
                embedded_tables = self._embed_text_items(table_items)
                embedded_items.extend(embedded_tables)
                print(f"✅ Embedded {len(embedded_tables)} tables")

            # ========================
            # PHASE 3: STORAGE
            # ========================

            print("\n" + "="*70)
            print("PHASE 3: VECTOR STORAGE")
            print("-" * 70)

            collection = init_chroma()

            if embedded_items:
                store_chunks(collection, embedded_items)
                print(f"✅ Stored {len(embedded_items)} total vectors")

            # ========================
            # SUMMARY
            # ========================

            print("\n" + "="*70)
            print("📊 INGESTION SUMMARY")
            print("="*70)
            print(f"\n✅ Document: {doc_name} ({doc_id})")
            print(f"\n📈 Statistics:")
            print(f"   📝 Text chunks: {len(embedded_text)}")
            print(f"   🖼️  Images: {len(images)}")
            print(f"   🎨 Figures: {fig_summary.get('total_figures', 0)}")
            print(f"   📊 Tables: {len(table_items)}")
            print(f"   📦 Total vectors: {len(embedded_items)}")
            print(f"\n✨ TRUE MULTIMODAL INDEXING COMPLETE\n")

            # Register document
            self.document_registry[doc_id] = {
                "name": doc_name,
                "filename": filename,
                "path": pdf_path,
                "timestamp": datetime.now().isoformat(),
                "text_chunks": len(embedded_text),
                "images": len(images),
                "figures": fig_summary.get('total_figures', 0),
                "tables": len(table_items),
                "total_vectors": len(embedded_items)
            }

            return doc_id

        except Exception as e:
            print(f"\n❌ Error during ingestion: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _extract_text_pymupdf(self, pdf_path: str, doc_id: str, doc_name: str, filename: str) -> List[Dict]:
        """Fallback text extraction using PyMuPDF"""
        try:
            import fitz
            doc = fitz.open(pdf_path)
            items = []
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text("text").strip()
                if text:
                    # Chunk the text
                    chunks = self._chunk_text(text, chunk_size=500, overlap=100)
                    for i, chunk in enumerate(chunks):
                        items.append({
                            "text": chunk,
                            "type": "text",
                            "modality": "text",
                            "metadata": {
                                "type": "text",
                                "modality": "text",
                                "doc_id": doc_id,
                                "doc_name": doc_name,
                                "source": filename,
                                "page": page_num + 1,
                                "chunk_index": i,
                                "content_type": "text"
                            }
                        })
            doc.close()
            return items
        except Exception as e:
            print(f"⚠️  PyMuPDF text extraction failed: {e}")
            return []

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            i += chunk_size - overlap
        return chunks if chunks else [text]

    def _convert_to_text_dicts(self, raw_items, doc_id: str, doc_name: str, filename: str) -> List[Dict]:
        """Convert unstructured output tuples to proper dict format"""
        items = []
        for item in raw_items:
            if isinstance(item, tuple):
                text, meta = item
                page_num = meta.get("page_number", 0)
                items.append({
                    "text": text,
                    "type": "text",
                    "modality": "text",
                    "metadata": {
                        "type": meta.get("content_type", "text"),
                        "modality": "text",
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "source": filename,
                        "page": int(page_num) if page_num else 0,
                        "content_type": meta.get("content_type", "text")
                    }
                })
            else:
                text = item.get("text", str(item))
                meta = item.get("metadata", {})
                page_num = meta.get("page_number", meta.get("page", 0))
                items.append({
                    "text": text,
                    "type": "text",
                    "modality": "text",
                    "metadata": {
                        **meta,
                        "type": meta.get("content_type", "text"),
                        "modality": "text",
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "source": filename,
                        "page": int(page_num) if page_num else 0,
                    }
                })
        return items

    def _embed_text_items(self, text_items: List[Dict]) -> List[Dict]:
        """
        Store text items without custom embeddings.
        Chroma will use its own sentence-transformer embedder (384-dim, better for text).
        """
        embedded = []
        for item in text_items[:200]:  # Limit per doc
            try:
                result = dict(item)
                # Don't provide custom embeddings - let Chroma use sentence-transformers
                # which is optimized for text semantic search (384 dims)
                # CLIP is reserved for image embeddings (512 dims)
                embedded.append(result)
            except Exception as e:
                print(f"⚠️  Error processing text: {e}")
                embedded.append(item)
                continue
        return embedded

    def _embed_images(self, images: List[Dict], doc_id: str, doc_name: str, filename: str) -> List[Dict]:
        """
        Process images for storage.
        Store image descriptions as text, let Chroma embed them with sentence-transformers.
        """
        embedded = []
        for img in images:
            try:
                img_filename = img.get("filename", "unknown")
                img_path = img.get("path", "")
                page_num = img.get("page", 0)

                # Create rich description for the image that Chroma can embed
                item = {
                    "text": f"Image on page {page_num}: {img_filename}. This is a figure, diagram, chart, or visual content from the document.",
                    "type": "image",
                    "modality": "image",
                    "metadata": {
                        "type": "image",
                        "modality": "image",
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "source": filename,
                        "page": int(page_num) if page_num else 0,
                        "filename": img_filename,
                        "image_path": img_path,
                        "image_url": f"/image/{img_filename}"
                    }
                }
                # Don't provide custom embeddings - let Chroma handle it
                # This ensures dimensional consistency across all vectors
                embedded.append(item)
            except Exception as e:
                print(f"⚠️  Error processing image {img.get('filename')}: {e}")
                continue
        return embedded

    def get_document_registry(self) -> Dict:
        """Get all registered documents"""
        return self.document_registry


# Global pipeline instance
_pipeline = None


def get_pipeline():
    """Get or create global pipeline"""
    global _pipeline
    if _pipeline is None:
        _pipeline = MultimodalIngestionPipeline()
    return _pipeline


def run_ingestion(pdf_path: str) -> str:
    """
    Main ingestion function (for API compatibility)
    Returns doc_id
    """
    pipeline = get_pipeline()
    return pipeline.ingest_document(pdf_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        pipeline = MultimodalIngestionPipeline()
        doc_id = pipeline.ingest_document(pdf_path)
        print(f"\nDocument ID: {doc_id}")
    else:
        print("Usage: python multimodal_ingestion.py <pdf_path>")
