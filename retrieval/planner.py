# retrieval/planner.py

def plan(query: str):
    q = query.lower()

    if "table" in q or "compare" in q or "results" in q:
        strategy = "table"

    elif "figure" in q or "diagram" in q:
        strategy = "figure"

    elif "explain" in q or "method" in q:
        strategy = "semantic"

    else:
        strategy = "hybrid"

    return {
        "query": query,
        "strategy": strategy
    }