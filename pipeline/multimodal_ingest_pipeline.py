"""
Enhanced multimodal ingestion pipeline
Processes PDFs and extracts: text chunks, images, tables
Then embeds all modalities for unified retrieval
"""

import os
from pathlib import Path
from typing import Dict, Optional
import json

from ingestion.multimodal_extractor import MultimodalExtractor
from embeddings.multimodal_embedder import MultimodalEmbedder


class MultimodalIngestionPipeline:
    """Extract, embed, and index multimodal content"""

    def __init__(self, use_clip: bool = True):
        """
        Initialize ingestion pipeline
        
        Args:
            use_clip: Whether to use CLIP for image embeddings
        """
        self.use_clip = use_clip
        self.embedder = None
        self.metadata_file = "multimodal_metadata.json"
        
        try:
            if use_clip:
                print("🤖 Initializing CLIP embedder...")
                self.embedder = MultimodalEmbedder()
        except ImportError as e:
            print(f"⚠️  CLIP not available: {e}")
            print("   Image embeddings will be skipped")

    def ingest_pdf(self, pdf_path: str) -> Dict:
        """
        Ingest PDF with full multimodal processing
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dict with extracted and embedded content
        """
        print(f"\n📄 Ingesting multimodal PDF: {pdf_path}")
        
        # Step 1: Extract content
        extractor = MultimodalExtractor(pdf_path)
        extracted = extractor.extract_all()
        
        if "error" in extracted:
            return {"error": extracted["error"]}
        
        # Step 2: Embed content
        if self.embedder:
            print("\n🧠 Embedding content...")
            
            # Embed text chunks
            if extracted.get("text_chunks"):
                print("   📝 Embedding text chunks...")
                embedded_texts = self.embedder.batch_embed_mixed([
                    {
                        "modality": "text",
                        "content": chunk["text"],
                        "metadata": {"page": chunk["page"], "modality": "text"}
                    }
                    for chunk in extracted["text_chunks"]
                ])
                extracted["text_chunks"] = embedded_texts
            
            # Embed images
            if extracted.get("images"):
                print("   🖼️  Embedding images...")
                embedded_images = self.embedder.batch_embed_mixed([
                    {
                        "modality": "image",
                        "content": img["path"],
                        "metadata": img
                    }
                    for img in extracted["images"]
                ])
                extracted["images"] = embedded_images
            
            # Embed tables
            if extracted.get("tables"):
                print("   📊 Embedding tables...")
                embedded_tables = self.embedder.batch_embed_mixed([
                    {
                        "modality": "table",
                        "content": table["content"],
                        "metadata": table
                    }
                    for table in extracted["tables"]
                ])
                extracted["tables"] = embedded_tables
        
        # Step 3: Save metadata
        self._save_metadata(pdf_path, extracted)
        
        print("\n✅ Multimodal ingestion complete!")
        print(f"   📝 Text chunks: {len(extracted.get('text_chunks', []))}")
        print(f"   🖼️  Images: {len(extracted.get('images', []))}")
        print(f"   📊 Tables: {len(extracted.get('tables', []))}")
        
        return extracted

    def _save_metadata(self, pdf_path: str, extracted: Dict):
        """Save metadata for later retrieval"""
        try:
            metadata = {
                "source_file": pdf_path,
                "num_pages": extracted["metadata"]["total_pages"],
                "num_text_chunks": len(extracted.get("text_chunks", [])),
                "num_images": len(extracted.get("images", [])),
                "num_tables": len(extracted.get("tables", [])),
                "images": [
                    {
                        "filename": img.get("filename"),
                        "page": img.get("page"),
                        "path": img.get("path"),
                        "embedding_dim": len(img.get("embedding", []))
                    }
                    for img in extracted.get("images", [])
                ],
                "tables": [
                    {
                        "page": table.get("page"),
                        "embedding_dim": len(table.get("embedding", []))
                    }
                    for table in extracted.get("tables", [])
                ]
            }
            
            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"   💾 Metadata saved to {self.metadata_file}")
        
        except Exception as e:
            print(f"   ⚠️  Could not save metadata: {e}")

    def load_metadata(self) -> Dict:
        """Load saved metadata"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load metadata: {e}")
        
        return {}


def test_pipeline():
    """Test multimodal ingestion"""
    pipeline = MultimodalIngestionPipeline(use_clip=True)
    
    # Test with a sample PDF if available
    test_pdf = "sample_paper.pdf"
    
    if os.path.exists(test_pdf):
        result = pipeline.ingest_pdf(test_pdf)
        
        if "error" not in result:
            print("\n📊 Ingestion Results:")
            print(f"  Text chunks: {len(result.get('text_chunks', []))}")
            print(f"  Images: {len(result.get('images', []))}")
            print(f"  Tables: {len(result.get('tables', []))}")
        else:
            print(f"Error: {result['error']}")
    else:
        print(f"Test PDF not found: {test_pdf}")


if __name__ == "__main__":
    test_pipeline()
