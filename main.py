# main.py - MULTIMODAL RAG SYSTEM

# CRITICAL: RUN_INGESTION Flag
# ===========================
# RUN_INGESTION = True:
#   - Use ONLY on FIRST RUN or when adding NEW PDFs
#   - Extracts text, images, tables from PDF files  
#   - Embeds all with CLIP (creates vector embeddings)
#   - Stores everything in Chroma vector database
#   - Takes TIME (slower) - don't set True on every run!
#
# RUN_INGESTION = False:
#   - Use for SUBSEQUENT RUNS after initial ingestion
#   - Skips extraction/embedding (FAST)
#   - Uses already-indexed documents in Chroma
#   - You can still ask questions and get answers
#   - Recommended for testing and demos
#
# WORKFLOW:
# 1. First time: Set RUN_INGESTION = True, run script
# 2. Subsequent times: Set RUN_INGESTION = False, run script  
# 3. Adding new PDF: Set RUN_INGESTION = True again

from pipeline.ingest_pipeline import run_ingestion
from vectordb.chroma_client import init_chroma
from retrieval.multimodal_pipeline import run_multimodal_rag

RUN_INGESTION = True  # SET TO TRUE FOR FIRST RUN - now fresh start!


def main():
    print("\n" + "="*70)
    print("MULTIMODAL RAG SYSTEM")
    print("="*70)
    
    # Phase 1: INGESTION (only on first run)
    if RUN_INGESTION:
        print("\nWARNING: RUN_INGESTION = True")
        print("Starting document ingestion...")
        run_ingestion()
        print("\nNext time, set RUN_INGESTION = False\n")
    else:
        print("\nRUN_INGESTION = False (using cached documents)\n")
    
    # Phase 2: Initialize database
    print("Initializing vector database...")
    collection = init_chroma()
    print("Ready!\n")
    
    # Phase 3: Test queries
    queries = [
        "What is the main topic?",
        "Explain key concepts",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        try:
            result = run_multimodal_rag(query)
            print(f"Answer: {result['answer'][:200]}...")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
