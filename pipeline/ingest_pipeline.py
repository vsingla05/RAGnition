# pipeline/ingest_pipeline.py

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import PDF_PATH
except ImportError:
    from backend.config import PDF_PATH

try:
    from ingestion.pdf_parser import extract_pdf_multimodal
    from ingestion.simple_pdf_parser import extract_pdf_simple
except ImportError:
    from backend.ingestion.pdf_parser import extract_pdf_multimodal
    from backend.ingestion.simple_pdf_parser import extract_pdf_simple

try:
    from vectordb.chroma_client import init_chroma, store_chunks, test_query
except ImportError:
    from backend.vectordb.chroma_client import init_chroma, store_chunks, test_query


def run_ingestion(pdf_path: str):

    print("🚀 Starting ingestion...\n")
    
    # Use the provided pdf_path, fallback to PDF_PATH from config
    file_to_ingest = pdf_path if pdf_path else PDF_PATH
    print(f"📄 Processing: {file_to_ingest}")

    try:
        # Try advanced extraction first
        items = extract_pdf_multimodal(file_to_ingest)
        
        if len(items) == 0:
            print("⚠️  Advanced extraction returned no chunks, trying simple extraction...")
            items = extract_pdf_simple(file_to_ingest)
        
        print(f"✅ Extracted {len(items)} chunks from PDF")
        
        if len(items) == 0:
            print("⚠️  No chunks extracted - the PDF might be empty or unreadable")
            return
        
        collection = init_chroma()
        store_chunks(collection, items)
        test_query(collection)
        
        print("\n🎯 Phase 1 complete ✅")
    except ImportError:
        # If unstructured is not available, fall back to simple parser
        print("⚠️  Advanced PDF library not available, using simple extraction...")
        try:
            items = extract_pdf_simple(file_to_ingest)
            print(f"✅ Extracted {len(items)} chunks from PDF")
            
            if len(items) == 0:
                print("⚠️  No chunks extracted - the PDF might be empty or unreadable")
                return
            
            collection = init_chroma()
            store_chunks(collection, items)
            test_query(collection)
            
            print("\n🎯 Phase 1 complete ✅")
        except Exception as e:
            print(f"❌ Error during PDF processing: {str(e)}")
            raise
    except Exception as e:
        print(f"❌ Error during PDF processing: {str(e)}")
        raise