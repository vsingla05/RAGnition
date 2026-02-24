# retrieval/query_analyzer.py

from typing import Dict


def analyze_query(query: str) -> Dict:
    """
    Very important step: Query intent detection.

    Determines:
    - query_type
    - metadata filters
    """

    q = query.lower()

    if any(k in q for k in ["table", "results", "statistics", "compare"]):
        query_type = "table"

    elif any(k in q for k in ["figure", "diagram", "architecture", "visual"]):
        query_type = "image_caption"

    else:
        query_type = "text"

    filters = {"content_type": query_type}

    return {
        "query": query,
        "query_type": query_type,
        "filters": filters
    }