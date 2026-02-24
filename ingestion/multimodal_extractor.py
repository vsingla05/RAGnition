"""
Multimodal Ingestion - Extract images, tables, and text from PDFs
Handles: figures, tables, diagrams, charts from documents
"""

import os
import base64
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from PIL import Image
except ImportError:
    Image = None


class MultimodalExtractor:
    """Extract text, images, tables from PDF with metadata"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.images_dir = Path("extracted_images")
        self.images_dir.mkdir(exist_ok=True)
        self.doc = None
        self.image_count = 0
        
    def extract_all(self) -> Dict:
        """Extract all modalities from PDF"""
        if not fitz:
            return {"error": "PyMuPDF not installed"}
        
        self.doc = fitz.open(self.pdf_path)
        
        results = {
            "text_chunks": [],
            "images": [],
            "tables": [],
            "metadata": {
                "total_pages": self.doc.page_count,
                "filename": Path(self.pdf_path).name,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        print(f"📄 Processing {self.doc.page_count} pages...")
        
        for page_num in range(self.doc.page_count):
            page = self.doc[page_num]
            
            # Extract text
            text = page.get_text("text")
            if text.strip():
                results["text_chunks"].append({
                    "text": text,
                    "page": page_num + 1,
                    "modality": "text"
                })
            
            # Extract images
            images = self._extract_images_from_page(page, page_num + 1)
            results["images"].extend(images)
            
            # Extract tables (as structured text for now)
            tables = self._extract_tables_from_page(page, page_num + 1)
            results["tables"].extend(tables)
        
        self.doc.close()
        
        print(f"✅ Extracted:")
        print(f"   📝 {len(results['text_chunks'])} text chunks")
        print(f"   🖼️  {len(results['images'])} images")
        print(f"   📊 {len(results['tables'])} tables")
        
        return results
    
    def _extract_images_from_page(self, page, page_num: int) -> List[Dict]:
        """Extract images from a page"""
        images = []
        
        try:
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(self.doc, xref)
                    
                    # Save image
                    img_filename = f"page_{page_num}_img_{img_index}.png"
                    img_path = self.images_dir / img_filename
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        pix.save(str(img_path))
                    else:  # RGBA
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                        pix.save(str(img_path))
                    
                    self.image_count += 1
                    
                    images.append({
                        "path": str(img_path),
                        "filename": img_filename,
                        "page": page_num,
                        "index": img_index,
                        "modality": "image",
                        "type": "figure"
                    })
                    
                    print(f"   Saved: {img_filename}")
                    
                except Exception as e:
                    print(f"   ⚠️  Error extracting image {img_index} from page {page_num}: {e}")
                    
        except Exception as e:
            print(f"   ⚠️  Error processing images on page {page_num}: {e}")
        
        return images
    
    def _extract_tables_from_page(self, page, page_num: int) -> List[Dict]:
        """Extract tables from a page"""
        tables = []
        
        try:
            # Find tables (basic detection)
            tabs = page.find_tables()
            
            for tab_index, tab in enumerate(tabs):
                try:
                    table_dict = tab.to_dict()
                    
                    tables.append({
                        "content": str(table_dict),
                        "page": page_num,
                        "index": tab_index,
                        "modality": "table",
                        "type": "table",
                        "raw_dict": table_dict
                    })
                    
                except Exception as e:
                    print(f"   ⚠️  Error extracting table {tab_index} from page {page_num}: {e}")
                    
        except Exception as e:
            print(f"   ⚠️  Error finding tables on page {page_num}: {e}")
        
        return tables
    
    def get_image_base64(self, image_path: str) -> str:
        """Convert image to base64 for API transmission"""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return ""


def test_extractor():
    """Test multimodal extractor"""
    pdf_path = "sample_paper.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ {pdf_path} not found")
        return
    
    extractor = MultimodalExtractor(pdf_path)
    results = extractor.extract_all()
    
    print("\n📊 Extraction Results:")
    print(f"Text chunks: {len(results['text_chunks'])}")
    print(f"Images: {len(results['images'])}")
    print(f"Tables: {len(results['tables'])}")
    
    if results['images']:
        print("\n🖼️  Images extracted:")
        for img in results['images'][:3]:
            print(f"  - {img['filename']} (Page {img['page']})")


if __name__ == "__main__":
    test_extractor()
