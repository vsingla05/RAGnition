# retrieval/self_improving_pipeline.py

try:
    from retrieval.generator import generate_answer
    from retrieval.multimodal_router import MultimodalQueryRouter
except ImportError:
    from generator import generate_answer
    from multimodal_router import MultimodalQueryRouter


def run_self_improving_retrieval(collection, query: str, max_attempts: int = 3, generate_final_answer: bool = True):
    """
    TRUE MULTIMODAL RETRIEVAL PIPELINE
    
    Searches BOTH text and images based on query intent.
    Uses CLIP embeddings in same space for cross-modal search.
    
    Args:
        collection: Chroma collection with text + image vectors
        query: User question
        max_attempts: Max retrieval attempts
        generate_final_answer: Whether to generate LLM answer
        
    Returns:
        Dictionary with answer, chunks, sources, and metadata
    """

    print(f"\n🔍 MULTIMODAL Retrieval: {query}")

    # STEP 1: Analyze query intent (NEW - MULTIMODAL)
    print(f"📊 Analyzing query intent...")
    router = MultimodalQueryRouter()
    analysis = router.analyze_query(query)
    print(f"   Primary modality: {analysis['primary_modality'].value}")
    print(f"   Confidence: {analysis['confidence']:.1%}")
    
    # STEP 2: Retrieve from vector DB (BOTH text and images)
    print(f"� Querying multimodal vector database...")
    raw = collection.query(query_texts=[query], n_results=15)
    all_docs = raw.get("documents", [[]])[0]
    all_metadatas = raw.get("metadatas", [[]])[0] if "metadatas" in raw else []
    
    print(f"✅ Retrieved {len(all_docs)} documents")
    
    # STEP 3: Separate text and images (NEW - MULTIMODAL)
    text_chunks = []
    image_chunks = []
    
    for i, doc in enumerate(all_docs[:10]):
        meta = all_metadatas[i] if i < len(all_metadatas) else {}
        
        if meta.get("modality") == "image" or "Image:" in doc:
            # This is an image
            image_chunks.append({
                "text": doc,
                "path": meta.get("path", ""),
                "filename": meta.get("filename", ""),
                "page": meta.get("page", ""),
                "metadata": meta,
            })
            print(f"   🖼️  Image: {meta.get('filename', 'unknown')}")
        else:
            # This is text
            text_chunks.append({
                "text": doc,
                "metadata": meta,
            })
            print(f"   📝 Text chunk")
    
    # STEP 4: Vision reasoning on retrieved images (NEW - MULTIMODAL)
    print(f"\n👁️  Vision Analysis on Retrieved Images...")
    if image_chunks:
        try:
            from retrieval.vision_generator import get_vision_generator
            vision_gen = get_vision_generator()
            
            for img_chunk in image_chunks[:2]:  # Analyze top 2 images
                if img_chunk["path"] and os.path.exists(img_chunk["path"]):
                    print(f"   Analyzing: {img_chunk['filename']}")
                    vision_analysis = vision_gen.analyze_image(img_chunk["path"])
                    
                    if vision_analysis.get("status") == "success":
                        img_chunk["vision_analysis"] = {
                            "caption": vision_analysis.get("caption", ""),
                            "description": vision_analysis.get("description", ""),
                        }
                        print(f"      Caption: {vision_analysis.get('caption', '')[:50]}...")
        except Exception as e:
            print(f"   ⚠️  Vision analysis skipped: {e}")
    
    # STEP 5: Combine all chunks for answer generation
    all_chunks = text_chunks + image_chunks
    
    # STEP 6: Generate answer with multimodal context (UPGRADED)
    if generate_final_answer and all_chunks:
        print(f"\n🤖 Generating multimodal answer...")
        
        # Build context with both text and image analysis
        context_text = ""
        for chunk in text_chunks:
            context_text += chunk["text"] + "\n"
        
        for chunk in image_chunks:
            if chunk.get("vision_analysis"):
                context_text += f"\n[IMAGE: {chunk['filename']}]\n"
                context_text += f"Caption: {chunk['vision_analysis'].get('caption', '')}\n"
                context_text += f"Description: {chunk['vision_analysis'].get('description', '')}\n"
        
        generation_result = generate_answer(query, [{"text": context_text}])
        answer = generation_result.get("answer", "No answer generated")
        sources = generation_result.get("sources", [])
    else:
        answer = "Answer generation disabled"
        sources = []

    print(f"\n✅ MULTIMODAL Retrieval Complete")
    print(f"   Text chunks: {len(text_chunks)}")
    print(f"   Images: {len(image_chunks)}")
    
    return {
        "response": answer,
        "source_chunks": all_chunks,
        "sources": sources,
        "query_used": query,
        "original_query": query,
        "text_sources": len(text_chunks),
        "image_sources": len(image_chunks),
        "confidence": analysis["confidence"],
        "modality": analysis['primary_modality'].value,
        "chunks_count": len(all_chunks)
    }

import os
