# retrieval/reranker.py

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def rerank(query: str, docs):
    """
    Lightweight reranker.
    """

    if not docs:
        return []

    q_emb = model.encode(query, convert_to_tensor=True)
    d_emb = model.encode(docs, convert_to_tensor=True)

    scores = util.cos_sim(q_emb, d_emb)[0]

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [r[0] for r in ranked]