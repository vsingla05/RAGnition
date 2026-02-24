# retrieval/query_rewriter.py

def rewrite_query(query: str, attempt: int):
    """
    Deterministic rewriting.
    (Later → LLM rewriting)
    """

    if attempt == 1:
        return query + " detailed explanation"

    if attempt == 2:
        return query + " research paper"

    return query + " methodology results figures tables"