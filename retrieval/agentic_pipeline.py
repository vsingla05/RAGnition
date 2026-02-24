# retrieval/agentic_pipeline.py

from retrieval.planner import plan
from retrieval.tools import semantic_tool, hybrid_tool, table_tool, figure_tool
from retrieval.retrieval_memory import RetrievalMemory


TOOLS = {
    "semantic": semantic_tool,
    "hybrid": hybrid_tool,
    "table": table_tool,
    "figure": figure_tool
}


def run_agentic_retrieval(collection, query: str, max_steps: int = 2, return_results: bool = False):

    print(f"\n🤖 Agentic Retrieval Query: {query}")

    memory = RetrievalMemory()

    final_results = None   # ⭐ store results for Phase 5

    for step in range(max_steps):

        # 1️⃣ Planner
        plan_out = plan(query)
        strategy = plan_out["strategy"]

        print(f"\nStep {step+1} → strategy: {strategy}")

        # 2️⃣ Execute tool
        tool = TOOLS[strategy]
        result = tool(collection, query)

        # ⭐ save results
        final_results = result

        memory.add({
            "step": step + 1,
            "strategy": strategy
        })

        # 3️⃣ Reflection (simple heuristic)
        if strategy != "hybrid":
            print("Reflection → broadening search")

            strategy = "hybrid"
            tool = TOOLS[strategy]

            result = tool(collection, query)

            # ⭐ overwrite with broader results
            final_results = result

            memory.add({
                "step": step + 1,
                "strategy": "hybrid"
            })

            break

    print("\n🧠 Execution Trace:")
    for h in memory.history:
        print(h)

    # ⭐ Phase 5 hook
    if return_results:
        return final_results