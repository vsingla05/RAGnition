# retrieval/adaptive_retriever.py

from typing import Dict


def adaptive_retrieve(collection, analysis: Dict, top_k: int = 8):
    """
    Adaptive retrieval using metadata filters.
    """

    query = analysis["query"]
    filters = analysis["filters"]

    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filters   # ⭐ critical
        )
    except Exception:
        # fallback → no filter
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )

    return results