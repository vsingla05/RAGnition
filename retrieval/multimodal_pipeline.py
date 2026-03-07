"""
END-TO-END MULTIMODAL RAG PIPELINE - FIXED
Query -> Retrieve (text/images/tables from CURRENT DOC ONLY) -> Generate Answer
"""

import sys
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from retrieval.multimodal_retriever import retrieve_multimodal
    from retrieval.multimodal_generator import generate_multimodal_answer
except ImportError:
    from multimodal_retriever import retrieve_multimodal
    from multimodal_generator import generate_multimodal_answer


def run_multimodal_rag(query: str, doc_id: str = None, top_k: int = 5) -> Dict:
    """
    Complete multimodal RAG pipeline

    1. Query -> 2. Retrieve (text/images/tables from doc_id only) -> 3. Generate

    Args:
        query: User question
        doc_id: Document ID to search (REQUIRED for preventing cross-doc contamination)
        top_k: Number of results per modality

    Returns:
        {
            "query": "...",
            "answer": "...",
            "sources": [...],
            "images": [...],
            "modalities": {"text": N, "tables": N, "images": N},
            "confidence": 0.0-1.0
        }
    """

    print("\n" + "="*70)
    print("MULTIMODAL RAG PIPELINE")
    if doc_id:
        print(f"Document ID: {doc_id}")
    else:
        print("⚠️  No doc_id provided — searching ALL documents")
    print("="*70)

    try:
        # STEP 1: RETRIEVE
        print("\n[1/2] RETRIEVAL PHASE")
        print("-" * 70)
        retrieved = retrieve_multimodal(query, doc_id, top_k)

        if retrieved.get("error"):
            return {
                "query": query,
                "answer": f"Retrieval failed: {retrieved['error']}",
                "error": retrieved["error"],
                "sources": [],
                "images_referenced": [],
                "modalities": {"text": 0, "tables": 0, "images": 0},
                "confidence": 0.0
            }

        total = retrieved.get("total_retrieved", 0)
        if total == 0:
            msg = retrieved.get("message", "No relevant content found in the document.")
            return {
                "query": query,
                "answer": f"No relevant information found for your question. {msg}",
                "sources": [],
                "images_referenced": [],
                "modalities": {"text": 0, "tables": 0, "images": 0},
                "confidence": 0.0,
                "doc_id": doc_id
            }

        # STEP 2: GENERATE
        print("\n[2/2] GENERATION PHASE")
        print("-" * 70)
        generation_result = generate_multimodal_answer(query, retrieved)

        # COMBINE RESULTS
        result = {
            "query": query,
            "answer": generation_result["answer"],
            "confidence": generation_result["confidence"],
            "precision": generation_result.get("precision", 0.0),
            "recall": generation_result.get("recall", 0.0),
            "tp": generation_result.get("tp", 0),
            "fp": generation_result.get("fp", 0),
            "fn": generation_result.get("fn", 0),
            "tn": generation_result.get("tn", 0),
            "sources": generation_result["sources"],
            "images_referenced": generation_result["images_referenced"],
            "modalities": generation_result["modalities"],
            "total_sources": generation_result["total_sources"],
            "doc_id": doc_id
        }

        print("\n" + "="*70)
        print("MULTIMODAL RAG COMPLETE")
        print("="*70)
        print(f"Answer length: {len(result['answer'])} chars")
        print(f"Sources: {result['total_sources']}")
        print(f"   Text: {result['modalities']['text']}")
        print(f"   Tables: {result['modalities']['tables']}")
        print(f"   Images: {result['modalities']['images']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("="*70)

        return result

    except Exception as e:
        print(f"\nPipeline error: {e}")
        import traceback
        traceback.print_exc()

        return {
            "query": query,
            "answer": f"Pipeline error: {e}",
            "error": str(e),
            "sources": [],
            "images_referenced": [],
            "modalities": {"text": 0, "tables": 0, "images": 0},
            "confidence": 0.0
        }
