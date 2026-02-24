from generation.prompt_builder import build_prompt
from generation.llm_client import generate_answer
from retrieval.agentic_pipeline import run_agentic_retrieval


def run_rag(collection, query):

    print("\n================ Phase 5: RAG Answer ================\n")

    # 1️⃣ retrieve
    results = run_agentic_retrieval(collection, query, return_results=True)

    contexts = [r["text"] for r in results]

    # 2️⃣ prompt
    prompt = build_prompt(query, contexts)

    # 3️⃣ generate
    answer = generate_answer(prompt)

    print("🧠 Final Answer:\n")
    print(answer)

    return answer