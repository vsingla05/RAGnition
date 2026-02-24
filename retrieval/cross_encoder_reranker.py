# retrieval/cross_encoder_reranker.py

from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def cross_rerank(query, docs, top_k=8):
    if not docs:
        return []

    pairs = [(query, d) for d in docs]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [r[0] for r in ranked[:top_k]]