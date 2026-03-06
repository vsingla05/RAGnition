# vectordb/chroma_client.py
# Fixed: handles both dict format and tuple format for store_chunks

import uuid
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Union

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
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def store_chunks(collection, items: List[Union[Dict, Tuple]]):
    """
    Store chunks in Chroma. Accepts either:
    - List of dicts: {"text": str, "metadata": dict, "embedding": list (optional)}
    - List of tuples: (text: str, metadata: dict)
    """

    ids = []
    docs = []
    metas = []
    embeddings = []
    has_custom_embeddings = False

    for item in items:
        if isinstance(item, dict):
            # Dict format from multimodal pipeline
            text = item.get("text", "")
            metadata = item.get("metadata", {})
            emb = item.get("embedding", None)

            # Resolve nested metadata if present
            if not metadata and "modality" in item:
                # Flat dict format
                metadata = {
                    "type": item.get("type", item.get("modality", "text")),
                    "modality": item.get("modality", "text"),
                    "page": item.get("page", 0),
                    "doc_id": item.get("doc_id", ""),
                    "doc_name": item.get("doc_name", ""),
                }
                if item.get("filename"):
                    metadata["filename"] = item["filename"]
                if item.get("path"):
                    metadata["image_path"] = item["path"]

        elif isinstance(item, tuple) and len(item) == 2:
            # Tuple format: (text, metadata)
            text, metadata = item
            emb = None
        else:
            print(f"⚠️  Skipping unrecognized item format: {type(item)}")
            continue

        if not text or not str(text).strip():
            continue

        # Clean metadata — Chroma only allows str/int/float/bool values
        clean_meta = {}
        for k, v in metadata.items():
            if v is None:
                clean_meta[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            elif isinstance(v, list):
                clean_meta[k] = str(v)
            else:
                clean_meta[k] = str(v)

        ids.append(str(uuid.uuid4()))
        docs.append(str(text))
        metas.append(clean_meta)

        if emb is not None and len(emb) > 0:
            embeddings.append(emb)
            has_custom_embeddings = True
        else:
            embeddings.append(None)

    if not docs:
        print("⚠️  No valid chunks to store — skipping")
        return

    # Batch insert
    try:
        if has_custom_embeddings and all(e is not None for e in embeddings):
            # All items have custom CLIP embeddings
            collection.add(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=embeddings
            )
        else:
            # Use collection's default sentence-transformer embeddings
            collection.add(
                ids=ids,
                documents=docs,
                metadatas=metas
            )

        print(f"✅ Stored {len(docs)} chunks in vector DB")

    except Exception as e:
        print(f"❌ Error storing chunks: {e}")
        # Try without custom embeddings as fallback
        try:
            collection.add(
                ids=ids,
                documents=docs,
                metadatas=metas
            )
            print(f"✅ Stored {len(docs)} chunks (using default embeddings)")
        except Exception as e2:
            print(f"❌ Fallback storage also failed: {e2}")
            raise


def delete_document(collection, doc_id: str):
    """Delete all chunks for a specific document"""
    try:
        results = collection.get(where={"doc_id": doc_id})
        if results and results.get("ids"):
            collection.delete(ids=results["ids"])
            print(f"✅ Deleted {len(results['ids'])} chunks for doc_id={doc_id}")
    except Exception as e:
        print(f"⚠️  Error deleting document {doc_id}: {e}")


def test_query(collection):
    results = collection.query(
        query_texts=["figure table diagram"],
        n_results=5
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    for i, (d, m) in enumerate(zip(docs, metas), 1):
        print("\nResult", i)
        print("-" * 40)
        print(d[:300])
        print("\nMetadata:", m)