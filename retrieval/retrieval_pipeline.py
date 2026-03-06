# retrieval/retrieval_pipeline.py
# Legacy retrieval pipeline - kept for backward compatibility
# For new code, use multimodal_pipeline.py instead

try:
    from retrieval.multimodal_pipeline import run_multimodal_rag
except ImportError:
    from multimodal_pipeline import run_multimodal_rag


def run_retrieval(collection, query: str, use_answer_generation: bool = True, doc_id: str = None):
    """
    Complete RAG pipeline - delegates to multimodal pipeline

    Args:
        collection: Chroma collection (not used - multimodal pipeline manages this)
        query: User question
        use_answer_generation: Whether to generate LLM answer (always generates)
        doc_id: Document ID to filter by (CRITICAL for per-doc isolation)

    Returns:
        Dictionary with answer, sources, and metadata
    """

    result = run_multimodal_rag(query, doc_id=doc_id, top_k=5)

    return {
        "response": result.get("answer", "No answer found"),
        "source_chunks": result.get("sources", []),
        "confidence": result.get("confidence", 0.0),
        "images_referenced": result.get("images_referenced", []),
    }
