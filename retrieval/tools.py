# retrieval/tools.py
# Using lazy imports to avoid circular dependencies

def semantic_tool(collection, query):
    from retrieval.advanced_pipeline import run_advanced_retrieval
    return {"tool": "semantic", "result": run_advanced_retrieval(collection, query)}


def hybrid_tool(collection, query):
    from retrieval.advanced_pipeline import run_advanced_retrieval
    return {"tool": "hybrid", "result": run_advanced_retrieval(collection, query)}


def table_tool(collection, query):
    from retrieval.advanced_pipeline import run_advanced_retrieval
    q = query + " table"
    return {"tool": "table", "result": run_advanced_retrieval(collection, q)}


def figure_tool(collection, query):
    from retrieval.advanced_pipeline import run_advanced_retrieval
    q = query + " figure"
    return {"tool": "figure", "result": run_advanced_retrieval(collection, q)}