"""
Answer Generation Module - Converts retrieved chunks into final answers using LLM

Supports multiple backends:
- Ollama (local)
- OpenAI (API)
- HuggingFace (API)
- Gemini (API) ⭐ recommended free option
"""

import os
import requests
from typing import List, Dict
from datetime import datetime

# ================= CONFIG =================

GENERATOR_TYPE = os.getenv("GENERATOR_TYPE", "ollama")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# ================= CONTEXT FORMAT =================

def format_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into context string"""

    if not chunks:
        return "No relevant documents found."

    context_parts = []

    for i, chunk in enumerate(chunks[:5], 1):
        text = chunk.get("text", chunk) if isinstance(chunk, dict) else chunk
        metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}

        source = metadata.get("source", "Unknown")
        page = metadata.get("page_number", "")

        if page:
            context_parts.append(f"[Source: {source}, Page {page}]\n{text}")
        else:
            context_parts.append(f"[Source: {source}]\n{text}")

    return "\n\n---\n\n".join(context_parts)


# ================= OLLAMA =================

def generate_with_ollama(query: str, chunks: List[Dict]) -> str:
    try:
        context = format_context(chunks)

        prompt = f"""
You are a helpful research assistant.

Answer ONLY using the provided context.
If answer not present, say you don't know.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.4,
            },
            timeout=180,
        )

        if response.status_code != 200:
            return "Ollama error."

        return response.json().get("response", "").strip()

    except Exception as e:
        return f"Ollama error: {str(e)}"


# ================= OPENAI =================

def generate_with_openai(query: str, chunks: List[Dict]) -> str:
    try:
        if not OPENAI_API_KEY:
            return "OPENAI_API_KEY not configured."

        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        context = format_context(chunks)

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # ✅ cheap model
            messages=[
                {"role": "system", "content": "Answer using only provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"},
            ],
            temperature=0.4,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"OpenAI error: {str(e)}"


# ================= GEMINI ⭐ =================

def generate_with_gemini(query: str, chunks: List[Dict]) -> str:
    try:
        if not GEMINI_API_KEY:
            return "GEMINI_API_KEY not configured."

        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)

        context = format_context(chunks)

        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
Answer using ONLY the context.

Context:
{context}

Question:
{query}
"""

        res = model.generate_content(prompt)
        return res.text

    except Exception as e:
        return f"Gemini error: {str(e)}"


# ================= HUGGINGFACE =================

def generate_with_huggingface(query: str, chunks: List[Dict]) -> str:
    try:
        if not HF_API_KEY:
            return "HF_API_KEY not configured."

        context = format_context(chunks)

        prompt = f"Context:\n{context}\n\nQuestion:\n{query}\nAnswer:"

        response = requests.post(
            "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf",
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={"inputs": prompt},
            timeout=120,
        )

        if response.status_code != 200:
            return "HuggingFace error."

        result = response.json()

        if isinstance(result, list) and result:
            return result[0]["generated_text"].split("Answer:")[-1].strip()

        return "HF generation failed."

    except Exception as e:
        return f"HuggingFace error: {str(e)}"


# ================= MAIN ROUTER =================

def generate_answer(query: str, chunks: List[Dict]) -> Dict:
    print("\n⭐ ANSWER GENERATION")
    print(f"Generator → {GENERATOR_TYPE}")
    print(f"Chunks → {len(chunks)}")

    if GENERATOR_TYPE == "ollama":
        answer = generate_with_ollama(query, chunks)

    elif GENERATOR_TYPE == "openai":
        answer = generate_with_openai(query, chunks)

    elif GENERATOR_TYPE == "huggingface":
        answer = generate_with_huggingface(query, chunks)

    elif GENERATOR_TYPE == "gemini":
        answer = generate_with_gemini(query, chunks)

    else:
        answer = "Unknown generator type."

    # Source extraction
    sources = []
    for chunk in chunks[:3]:
        if isinstance(chunk, dict):
            metadata = chunk.get("metadata", {})
            sources.append({
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page_number", "N/A")
            })

    return {
        "answer": answer,
        "sources": sources,
        "generator": GENERATOR_TYPE,
        "timestamp": datetime.now().isoformat(),
    }


# ================= TEST =================

def test_generator():
    test_query = "What is machine learning?"

    test_chunks = [{
        "text": "Machine learning is a subset of AI that learns from data.",
        "metadata": {"source": "test.pdf", "page_number": 1}
    }]

    print(generate_answer(test_query, test_chunks))


if __name__ == "__main__":
    test_generator()