# vectordb/chroma_client.py

import uuid
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import chromadb
from chromadb.utils import embedding_functions

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL
except ImportError:
    from backend.config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL


def init_chroma():

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    return collection


def store_chunks(collection, items: List[Tuple[str, Dict]]):

    ids = []
    docs = []
    metas = []

    for text, meta in items:
        if not text.strip():
            continue

        ids.append(str(uuid.uuid4()))
        docs.append(text)
        metas.append(meta)

    if not docs:
        raise ValueError("No valid chunks to store")

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas
    )

    print(f"✅ Stored {len(docs)} chunks")


def test_query(collection):

    results = collection.query(
        query_texts=["figure table"],
        n_results=5
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    for i, (d, m) in enumerate(zip(docs, metas), 1):
        print("\nResult", i)
        print("-" * 40)
        print(d[:500])
        print("\nMetadata:", m)