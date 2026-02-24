# ingestion/pdf_parser.py

from typing import List, Tuple, Dict
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from unstructured.partition.pdf import partition_pdf
except ImportError:
    print("⚠️  unstructured not available, will use fallback parser")
    partition_pdf = None


def extract_pdf_multimodal(pdf_path: str) -> List[Tuple[str, Dict]]:
    """
    Robust academic PDF parsing using unstructured.
    Handles:
    - multi column
    - figures
    - tables
    - captions
    """
    
    if not partition_pdf:
        raise ImportError("unstructured library not available")
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not os.path.getsize(pdf_path) > 0:
        raise ValueError(f"PDF file is empty: {pdf_path}")
    
    print(f"📖 Parsing PDF (advanced): {pdf_path}")

    try:
        elements = partition_pdf(
            filename=str(pdf_path),
            infer_table_structure=True,
            strategy="hi_res"   # ⭐ important for better extraction
        )
    except Exception as e:
        print(f"⚠️  HI_RES strategy failed, trying auto strategy: {str(e)}")
        elements = partition_pdf(
            filename=str(pdf_path),
            infer_table_structure=True,
            strategy="auto"
        )

    results = []

    for el in elements:

        text = el.text
        if not text or not text.strip():
            continue

        # classify
        if el.category == "Table":
            content_type = "table"
        elif el.category == "FigureCaption":
            content_type = "image_caption"
        else:
            content_type = "text"

        metadata = {
            "content_type": content_type,
            "page_number": getattr(el.metadata, 'page_number', 0),
            "source": os.path.basename(pdf_path)
        }

        results.append((text, metadata))
        
    print(f"✅ Successfully extracted {len(results)} elements from PDF")

    return results