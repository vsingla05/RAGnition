# retrieval/confidence.py

def compute_confidence(scores):
    if not scores:
        return 0.0

    return sum(scores) / len(scores)
