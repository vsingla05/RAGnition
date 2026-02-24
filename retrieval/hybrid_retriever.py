# retrieval/hybrid_retriever.py

from collections import defaultdict
import math


def simple_bm25_score(query, docs):
    """
    Lightweight BM25-style lexical scoring.
    """

    q_terms = query.lower().split()
    scores = []

    for d in docs:
        d_terms = d.lower().split()

        score = 0
        for t in q_terms:
            tf = d_terms.count(t)
            if tf > 0:
                score += 1 + math.log(tf)

        scores.append(score)

    return scores


def hybrid_retrieve(collection, expanded_queries, top_k=8):
    """
    Vector retrieval for each query → merge.
    """

    merged_docs = []
    merged_meta = []

    for q in expanded_queries:
        res = collection.query(query_texts=[q], n_results=top_k)

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        merged_docs.extend(docs)
        merged_meta.extend(metas)

    # deduplicate
    unique = {}
    for d, m in zip(merged_docs, merged_meta):
        if d not in unique:
            unique[d] = m

    docs = list(unique.keys())
    metas = list(unique.values())

    return docs, metas