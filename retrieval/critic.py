# retrieval/critic.py

def evaluate_retrieval(query: str, docs: list):
    """
    Very lightweight critic.

    Returns:
    - score
    - decision (good / retry)
    """

    if not docs:
        return {"score": 0.0, "decision": "retry"}

    q_words = set(query.lower().split())

    overlap_scores = []
    for d in docs[:5]:
        d_words = set(d.lower().split())
        overlap = len(q_words & d_words) / (len(q_words) + 1)
        overlap_scores.append(overlap)

    score = sum(overlap_scores) / len(overlap_scores)

    decision = "good" if score > 0.15 else "retry"

    return {
        "score": score,
        "decision": decision
    }