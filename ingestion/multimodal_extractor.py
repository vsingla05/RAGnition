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
        # Use absolute path relative to this file so images always go to backend/extracted_images/
        self.images_dir = Path(__file__).parent.parent / "extracted_images"
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
        """
        Extract images from a page.
        
        Strategy:
        1. Extract embedded raster images but skip tiny ones (logos/icons).
        2. If significant vector drawings are detected or no large images exist, render the full page to capture diagrams/charts.
        """
        images = []
        
        try:
            image_list = page.get_images()
            valid_images_found = 0
            
            # Strategy 1: Extract embedded images
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    # Get image info to check dimensions
                    img_info = self.doc.extract_image(xref)
                    if img_info:
                        w, h = img_info.get("width", 0), img_info.get("height", 0)
                        # Skip small images (logos, icons, lines)
                        if w < 100 or h < 100:
                            continue

                    pix = fitz.Pixmap(self.doc, xref)
                    img_filename = f"page_{page_num}_img_{img_index}.png"
                    img_path = self.images_dir / img_filename
                    
                    if pix.n - pix.alpha < 4:
                        pix.save(str(img_path))
                    else:
                        pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                        pix_rgb.save(str(img_path))
                        pix_rgb = None
                    pix = None
                    
                    self.image_count += 1
                    valid_images_found += 1
                    
                    images.append({
                        "path": str(img_path),
                        "filename": img_filename,
                        "page": page_num,
                        "index": img_index,
                        "modality": "image",
                        "type": "embedded_image"
                    })
                    print(f"   Saved embedded image: {img_filename}")
                except Exception as e:
                    print(f"   ⚠️  Error extracting image {img_index}: {e}")

            # Strategy 2: Check for vector graphics/drawings
            drawings = page.get_drawings()
            
            # If no valid large images were found, OR there are significant drawings (often charts/diagrams)
            if valid_images_found == 0 or len(drawings) > 10:
                text_len = len(page.get_text("text").strip())
                # Don't render pages that are basically just long text with a tiny drawing (like a line)
                # But if there are lots of drawings (e.g., > 20) or very little text, render it.
                if len(drawings) > 20 or (valid_images_found == 0 and text_len < 1200 and len(drawings) > 0):
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    img_filename = f"page_{page_num}_rendered.png"
                    img_path = self.images_dir / img_filename
                    
                    pix.save(str(img_path))
                    pix = None
                    self.image_count += 1
                    
                    images.append({
                        "path": str(img_path),
                        "filename": img_filename,
                        "page": page_num,
                        "index": 999,
                        "modality": "image",
                        "type": "page_render",
                        "description": f"Page {page_num} rendered to capture vector shapes/diagrams"
                    })
                    print(f"   Saved page render: {img_filename} (captured vector/visual content)")

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
