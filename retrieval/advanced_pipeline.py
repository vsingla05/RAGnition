# retrieval/advanced_pipeline.py

from retrieval.query_analyzer import analyze_query
from retrieval.query_expander import expand_query
from retrieval.hybrid_retriever import hybrid_retrieve
from retrieval.cross_encoder_reranker import cross_rerank


def run_advanced_retrieval(collection, query: str):

    print(f"\n🚀 Advanced Retrieval Query: {query}")

    # 1️⃣ Query analysis
    analysis = analyze_query(query)

    # 2️⃣ Query expansion
    expanded = expand_query(query)

    # 3️⃣ Hybrid retrieval
    docs, metas = hybrid_retrieve(collection, expanded)

    # 4️⃣ Cross encoder rerank
    ranked_docs = cross_rerank(query, docs)

    print("\n📊 Advanced Explainable Retrieval:\n")

    for i, doc in enumerate(ranked_docs[:5]):

        idx = docs.index(doc)
        meta = metas[idx]

        trace = {
            "query_type": analysis["query_type"],
            "strategy": "hybrid + cross-encoder",
            "metadata": meta
        }

        print(f"\nResult {i+1}")
        print("-" * 50)
        print(doc[:500])
        print("\nTrace:", trace)