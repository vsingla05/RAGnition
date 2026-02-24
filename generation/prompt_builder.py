def build_prompt(question: str, contexts: list):

    context_text = "\n\n".join(contexts)

    prompt = f"""
You are an assistant answering using provided context only.

QUESTION:
{question}

CONTEXT:
{context_text}

Answer using context only.
If answer not found, say you don't know.
"""

    return prompt