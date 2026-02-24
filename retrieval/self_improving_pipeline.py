# retrieval/self_improving_pipeline.py

try:
    from retrieval.generator import generate_answer
except ImportError:
    from generator import generate_answer


def run_self_improving_retrieval(collection, query: str, max_attempts: int = 3, generate_final_answer: bool = True):
    """
    Simplified retrieval with answer generation
    
    Args:
        collection: Chroma collection
        query: User question
        max_attempts: Max retrieval attempts (unused - kept for compatibility)
        generate_final_answer: Whether to generate LLM answer
        
    Returns:
        Dictionary with answer, chunks, sources, and metadata
    """

    print(f"\n🔍 Retrieval Pipeline: {query}")

    # Simple vector similarity retrieval
    print(f"📊 Querying vector database...")
    raw = collection.query(query_texts=[query], n_results=10)
    docs = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0] if "metadatas" in raw else []
    
    print(f"✅ Retrieved {len(docs)} documents")

    # Prepare chunks with metadata
    chunks = []
    for i, doc in enumerate(docs[:5]):
        meta = metadatas[i] if i < len(metadatas) else {}
        
        chunks.append({
            "text": doc,
            "metadata": meta,
        })

    # Phase 5: Answer generation with LLM
    if generate_final_answer and chunks:
        print(f"🤖 Generating answer with LLM...")
        generation_result = generate_answer(query, chunks)
        answer = generation_result.get("answer", "No answer generated")
        sources = generation_result.get("sources", [])
    else:
        answer = "Answer generation disabled"
        sources = []

    return {
        "response": answer,
        "source_chunks": chunks,
        "sources": sources,
        "query_used": query,
        "original_query": query,
        "confidence": 0.75,
        "attempt": 1,
        "chunks_count": len(chunks)
    }
