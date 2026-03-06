"""
COMPLETE MULTIMODAL RETRIEVAL - FIXED
Retrieves and returns: Text + Images + Tables
Strictly filters by doc_id to prevent cross-document contamination
"""

import sys
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from vectordb.chroma_client import init_chroma
    from retrieval.vision_generator import get_vision_generator
except ImportError:
    from backend.vectordb.chroma_client import init_chroma
    from backend.retrieval.vision_generator import get_vision_generator


class MultimodalRetriever:
    """Retrieve all modalities: text, images, tables — with strict doc isolation"""

    def __init__(self, doc_id: str = None):
        self.collection = init_chroma()
        self.vision_gen = get_vision_generator()
        self.doc_id = doc_id

    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        """
        Retrieve all relevant modalities

        Returns:
        {
            "text_results": [...],
            "image_results": [...],
            "table_results": [...],
            "all_items": [...]  # Combined for LLM
        }
        """

        print(f"\n🔍 MULTIMODAL RETRIEVAL: {query}")
        print(f"📋 Doc filter: {self.doc_id or 'ALL DOCUMENTS'}")
        print("-" * 70)

        try:
            # Build where clause for strict doc filtering
            where_clause = None
            if self.doc_id:
                where_clause = {"doc_id": self.doc_id}

            # Query vector DB — get extra results to filter by modality
            n_results = min(top_k * 4, 20)
            
            query_kwargs = {
                "query_texts": [query],
                "n_results": n_results,
            }
            if where_clause:
                query_kwargs["where"] = where_clause

            print(f"🔎 Searching vector database (n={n_results})...")
            
            try:
                raw_results = self.collection.query(**query_kwargs)
            except Exception as e:
                if "no documents" in str(e).lower() or "0 results" in str(e).lower():
                    print("⚠️  No documents in collection")
                    return self._empty_result("No documents found. Please upload a PDF first.")
                raise

            documents = raw_results.get("documents", [[]])[0]
            metadatas = raw_results.get("metadatas", [[]])[0]
            ids = raw_results.get("ids", [[]])[0]
            distances = raw_results.get("distances", [[]])[0]

            if not documents:
                return self._empty_result("No relevant content found for your query.")

            # Separate by modality
            text_results = []
            image_results = []
            table_results = []

            for doc, meta, doc_id_val, distance in zip(documents, metadatas, ids, distances):
                # Get modality/type from metadata (handle various field names)
                item_type = (
                    meta.get("type") or
                    meta.get("modality") or
                    meta.get("content_type") or
                    "text"
                )
                similarity = max(0, 1 - distance)

                if item_type == "image":
                    image_results.append({
                        "type": "image",
                        "filename": meta.get("filename", ""),
                        "page": meta.get("page", 0),
                        "path": meta.get("image_path", meta.get("path", "")),
                        "image_url": meta.get("image_url", ""),
                        "doc_name": meta.get("doc_name", ""),
                        "similarity": similarity,
                        "id": doc_id_val
                    })
                elif item_type in ("table", "table_markdown"):
                    table_results.append({
                        "type": "table",
                        "page": meta.get("page", 0),
                        "summary": meta.get("summary", ""),
                        "content": doc,
                        "doc_name": meta.get("doc_name", ""),
                        "similarity": similarity,
                        "id": doc_id_val
                    })
                else:  # text, image_caption, etc.
                    text_results.append({
                        "type": "text",
                        "content": doc,
                        "page": meta.get("page", 0),
                        "doc_name": meta.get("doc_name", ""),
                        "source": meta.get("source", ""),
                        "similarity": similarity,
                        "id": doc_id_val
                    })

            # Trim to top_k per modality
            text_results = text_results[:top_k]
            image_results = image_results[:top_k]
            table_results = table_results[:top_k]

            print(f"✅ Retrieved:")
            print(f"   📝 {len(text_results)} text chunks")
            print(f"   🖼️  {len(image_results)} images")
            print(f"   📊 {len(table_results)} tables")

            # Analyze images with Vision API
            if image_results:
                print(f"\n👁️  Analyzing {len(image_results)} images with Vision API...")
                for img in image_results:
                    try:
                        img_path = img.get("path", "")
                        if img_path and Path(img_path).exists():
                            analysis = self.vision_gen.analyze_image(img_path)
                            img["caption"] = analysis.get("caption", "")
                            img["description"] = analysis.get("description", "")
                            img["key_elements"] = analysis.get("key_elements", "")
                        else:
                            img["caption"] = f"Figure on page {img.get('page', '?')}"
                            img["description"] = ""
                            if img_path:
                                print(f"   ⚠️  Image file not found: {img_path}")
                    except Exception as e:
                        print(f"   ⚠️  Error analyzing image: {e}")
                        img["caption"] = f"Image on page {img.get('page', '?')}"
                        img["description"] = ""

            # Build image URL for frontend
            for img in image_results:
                filename = img.get("filename", "")
                if filename:
                    img["url"] = f"/image/{filename}"
                else:
                    path = img.get("path", "")
                    if path:
                        img["url"] = f"/image/{Path(path).name}"

            # Combine for LLM context
            all_items = []

            for item in text_results:
                all_items.append({
                    "type": "text",
                    "content": item["content"],
                    "page": item["page"],
                    "source": f"Page {item['page']}",
                    "doc_name": item.get("doc_name", "")
                })

            for item in image_results:
                all_items.append({
                    "type": "image",
                    "path": item.get("path", ""),
                    "url": item.get("url", ""),
                    "caption": item.get("caption", ""),
                    "description": item.get("description", ""),
                    "key_elements": item.get("key_elements", ""),
                    "page": item["page"],
                    "source": f"Figure on page {item['page']}",
                    "doc_name": item.get("doc_name", "")
                })

            for item in table_results:
                all_items.append({
                    "type": "table",
                    "content": item["content"],
                    "summary": item.get("summary", ""),
                    "page": item["page"],
                    "source": f"Table on page {item['page']}",
                    "doc_name": item.get("doc_name", "")
                })

            return {
                "text_results": text_results,
                "image_results": image_results,
                "table_results": table_results,
                "all_items": all_items,
                "total_retrieved": len(all_items)
            }

        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_result(str(e), error=True)

    def _empty_result(self, message: str = "", error: bool = False) -> Dict:
        return {
            "text_results": [],
            "image_results": [],
            "table_results": [],
            "all_items": [],
            "total_retrieved": 0,
            **({"error": message} if error else {"message": message})
        }


def retrieve_multimodal(query: str, doc_id: str = None, top_k: int = 5) -> Dict:
    """Main retrieval function — always pass doc_id for isolation"""
    retriever = MultimodalRetriever(doc_id)
    return retriever.retrieve(query, top_k)
