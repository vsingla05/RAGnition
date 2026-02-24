# main.py

"""
Entry point for Query-Adaptive + Explainable RAG system.

Phases:
1. Ingestion
2. Adaptive Retrieval
3. Advanced Retrieval
4. Agentic Retrieval
5. Self-Improving Retrieval
"""

from pipeline.ingest_pipeline import run_ingestion
from vectordb.chroma_client import init_chroma

from retrieval.retrieval_pipeline import run_retrieval
from retrieval.advanced_pipeline import run_advanced_retrieval
from retrieval.agentic_pipeline import run_agentic_retrieval
from retrieval.self_improving_pipeline import run_self_improving_retrieval


# 🔑 IMPORTANT — set True only first run
RUN_INGESTION = False


# ---------------- Phase 2 + 3 ----------------
def demo_queries(collection):

    queries = [
        "What does the figure show?",
        "Show the results table",
        "Explain the methodology"
    ]

    print("\n================ Phase 2: Adaptive Retrieval ================\n")
    for q in queries:
        run_retrieval(collection, q)

    print("\n================ Phase 3: Advanced Retrieval ================\n")
    for q in queries:
        run_advanced_retrieval(collection, q)


# ---------------- Phase 4 ----------------
def demo_agentic(collection):

    queries = [
        "What does the figure show?",
        "Compare the results",
        "Explain methodology"
    ]

    print("\n================ Phase 4: Agentic Retrieval ================\n")

    for q in queries:
        run_agentic_retrieval(collection, q)


# ---------------- Phase 5 ----------------
def demo_self_improving(collection):

    queries = [
        "What does the figure show?",
        "Explain the dataset",
        "How are tables extracted?"
    ]

    print("\n================ Phase 5: Self-Improving Retrieval ================\n")

    for q in queries:
        run_self_improving_retrieval(collection, q)


# ---------------- MAIN ----------------
def main():

    # Phase 1 — ingestion (run once only)
    if RUN_INGESTION:
        print("\n🚀 Running ingestion (one-time step)\n")
        run_ingestion()

    # Init vector DB
    collection = init_chroma()

    # Phase 2 + 3
    demo_queries(collection)

    # Phase 4
    demo_agentic(collection)

    # Phase 5
    demo_self_improving(collection)


if __name__ == "__main__":
    main()