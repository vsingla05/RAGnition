# retrieval/query_expander.py

def expand_query(query: str):
    """
    Simple deterministic expansion.
    (Later → LLM expansion)
    """

    expansions = [
        query,
        f"{query} explanation",
        f"{query} details",
        f"{query} summary"
    ]

    return list(set(expansions))