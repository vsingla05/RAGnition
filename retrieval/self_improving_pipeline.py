# retrieval/self_improving_pipeline.py

from retrieval.critic import evaluate_retrieval
from retrieval.query_rewriter import rewrite_query
from retrieval.advanced_pipeline import run_advanced_retrieval
from retrieval.query_analyzer import analyze_query
from retrieval.query_expander import expand_query
from retrieval.hybrid_retriever import hybrid_retrieve
from retrieval.cross_encoder_reranker import cross_rerank

try:
    from retrieval.generator import generate_answer
except ImportError:
    from generator import generate_answer
    from generator import generate_answer


def run_self_improving_retrieval(collection, query: str, max_attempts: int = 3, generate_final_answer: bool = True):
    """
    Self-improving retrieval with answer generation
    
    Args:
        collection: Chroma collection
        query: User question
        max_attempts: Max query rewriting attempts
        generate_final_answer: Whether to generate LLM answer
        
    Returns:
        Dictionary with answer, chunks, sources, and metadata
    """

    print(f"\n🧠 Self-Improving Retrieval Query: {query}")

    current_query = query
    best_docs = None
    best_metas = None

    for attempt in range(1, max_attempts + 1):

        print(f"\nAttempt {attempt} → query: {current_query}")

        # Query retrieval
        raw = collection.query(query_texts=[current_query], n_results=8)
        docs = raw.get("documents", [[]])[0]
        
        # Get metadatas
        metadatas = raw.get("metadatas", [[]])[0] if "metadatas" in raw else []

        # Critic evaluation
        critique = evaluate_retrieval(current_query, docs)
        print("Critic:", critique)

        if critique["decision"] == "good":
            print("✅ Retrieval accepted")
            best_docs = docs
            best_metas = metadatas
            break

        # Retry with rewritten query
        if attempt < max_attempts:
            current_query = rewrite_query(query, attempt)
            print("🔁 Rewriting query →", current_query)
        else:
            print("⚠️ Max attempts reached — returning best available")
            best_docs = docs
            best_metas = metadatas

    # Advanced retrieval for ranking
    analysis = analyze_query(current_query)
    expanded = expand_query(current_query)
    retrieved_docs, retrieved_metas = hybrid_retrieve(collection, expanded)
    ranked_docs = cross_rerank(current_query, retrieved_docs)

    print("\n📊 Advanced Explainable Retrieval:\n")
    
    chunks = []
    for i, doc in enumerate(ranked_docs[:5]):
        idx = retrieved_docs.index(doc) if doc in retrieved_docs else i
        meta = retrieved_metas[idx] if idx < len(retrieved_metas) else {}

        trace = {
            "query_type": analysis["query_type"],
            "strategy": "hybrid + cross-encoder",
            "metadata": meta
        }

        print(f"\nResult {i+1}")
        print("-" * 50)
        print(doc[:500])
        print("\nTrace:", trace)
        
        # Store chunk with metadata
        chunks.append({
            "text": doc,
            "metadata": meta,
            "trace": trace
        })

    # ⭐ PHASE 5: Answer generation
    if generate_final_answer and chunks:
        generation_result = generate_answer(current_query, chunks)
        answer = generation_result.get("answer", "No answer generated")
        sources = generation_result.get("sources", [])
    else:
        answer = "Answer generation disabled or no chunks retrieved"
        sources = []

    return {
        "response": answer,
        "source_chunks": chunks,
        "sources": sources,
        "query_used": current_query,
        "original_query": query,
        "confidence": critique.get("confidence", 0.5),
        "attempt": attempt,
        "chunks_count": len(chunks)
    }