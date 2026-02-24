# ingestion/simple_pdf_parser.py

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None


def extract_pdf_simple(pdf_path: str) -> List[Tuple[str, Dict]]:
    """
    Simple PDF extraction using PyPDF2 as fallback.
    Handles basic text extraction from PDFs.
    """
    
    if not PdfReader:
        raise ImportError("PyPDF2 library not available")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not os.path.getsize(pdf_path) > 0:
        raise ValueError(f"PDF file is empty: {pdf_path}")
    
    print(f"📖 Extracting PDF (simple method): {pdf_path}")
    
    results = []
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            num_pages = len(reader.pages)
            print(f"📄 PDF has {num_pages} pages")
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                
                if text and text.strip():
                    metadata = {
                        "content_type": "text",
                        "page_number": page_num,
                        "source": os.path.basename(pdf_path)
                    }
                    results.append((text, metadata))
        
        print(f"✅ Extracted {len(results)} pages from PDF")
        return results
        
    except Exception as e:
        print(f"❌ Error extracting PDF: {str(e)}")
        raise
