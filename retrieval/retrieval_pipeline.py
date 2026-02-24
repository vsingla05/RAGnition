# retrieval/retrieval_pipeline.py

try:
    from retrieval.self_improving_pipeline import run_self_improving_retrieval
except ImportError:
    from self_improving_pipeline import run_self_improving_retrieval


def run_retrieval(collection, query: str, use_answer_generation: bool = True):
    """
    Complete RAG pipeline with retrieval and answer generation
    
    Args:
        collection: Chroma collection
        query: User question
        use_answer_generation: Whether to generate LLM answer (Phase 5)
        
    Returns:
        Dictionary with answer, sources, and metadata
    """

    print(f"\n{'='*60}")
    print(f"� RAG PIPELINE START")
    print(f"{'='*60}")
    print(f"Query: {query}")
    print(f"Answer Generation: {use_answer_generation}")
    print(f"{'='*60}\n")

    # Use self-improving retrieval with answer generation
    result = run_self_improving_retrieval(
        collection, 
        query, 
        max_attempts=3,
        generate_final_answer=use_answer_generation
    )

    print(f"\n{'='*60}")
    print(f"🎯 RAG PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Final Answer Generated: {len(result.get('response', ''))} characters")
    print(f"📊 Chunks Retrieved: {result.get('chunks_count', 0)}")
    print(f"🎯 Confidence: {result.get('confidence', 0):.2%}")
    print(f"{'='*60}\n")

    return result