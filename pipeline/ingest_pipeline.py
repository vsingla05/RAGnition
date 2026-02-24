# pipeline/ingest_pipeline.py

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import PDF_PATH
except ImportError:
    from backend.config import PDF_PATH

try:
    from ingestion.pdf_parser import extract_pdf_multimodal
    from ingestion.multimodal_extractor import MultimodalExtractor
except ImportError:
    from backend.ingestion.pdf_parser import extract_pdf_multimodal
    from backend.ingestion.multimodal_extractor import MultimodalExtractor

try:
    from vectordb.chroma_client import init_chroma, store_chunks, test_query
except ImportError:
    from backend.vectordb.chroma_client import init_chroma, store_chunks, test_query

try:
    from embeddings.multimodal_embedder import MultimodalEmbedder
except ImportError:
    from backend.embeddings.multimodal_embedder import MultimodalEmbedder


def run_ingestion(pdf_path: str):
    """
    TRUE MULTIMODAL INGESTION
    Extracts text AND images, embeds with CLIP, stores both in vector DB
    """
    
    print("🚀 Starting MULTIMODAL ingestion...\n")
    
    # Use the provided pdf_path, fallback to PDF_PATH from config
    file_to_ingest = pdf_path if pdf_path else PDF_PATH
    print(f"📄 Processing: {file_to_ingest}")

    try:
        # STEP 1: Extract text chunks (original)
        print("\n📝 Extracting text chunks...")
        text_items = extract_pdf_multimodal(file_to_ingest)
        print(f"✅ Extracted {len(text_items)} text chunks")
        
        # STEP 2: Extract images (NEW - MULTIMODAL)
        print("\n🖼️  Extracting images...")
        extractor = MultimodalExtractor(file_to_ingest)
        extracted_data = extractor.extract_all()
        images = extracted_data.get("images", [])
        print(f"✅ Extracted {len(images)} images")
        
        # STEP 3: Initialize CLIP embedder (NEW - MULTIMODAL)
        print("\n🤖 Initializing CLIP embedder...")
        embedder = MultimodalEmbedder()
        
        # STEP 4: Embed text with CLIP (UPGRADED - was using MiniLM, now CLIP)
        print("\n🧠 Embedding text with CLIP...")
        embedded_text_items = []
        for item in text_items[:50]:  # Limit for speed
            try:
                emb = embedder.embed_text(item["text"])
                embedded_text_items.append({
                    **item,
                    "embedding": emb[0].tolist() if len(emb) > 0 else [],
                    "modality": "text"
                })
            except Exception as e:
                print(f"⚠️  Error embedding text: {e}")
                continue
        print(f"✅ Embedded {len(embedded_text_items)} text chunks")
        
        # STEP 5: Embed images with CLIP (NEW - MULTIMODAL)
        print("\n🖼️  Embedding images with CLIP...")
        embedded_images = []
        for img in images:
            try:
                emb = embedder.embed_image(img["path"])
                embedded_images.append({
                    "text": f"Image: {img.get('filename', 'unknown')}",
                    "embedding": emb[0].tolist() if len(emb) > 0 else [],
                    "path": img["path"],
                    "filename": img.get("filename"),
                    "page": img.get("page"),
                    "modality": "image",
                    "type": "figure"
                })
            except Exception as e:
                print(f"⚠️  Error embedding image {img.get('filename')}: {e}")
                continue
        print(f"✅ Embedded {len(embedded_images)} images")
        
        # STEP 6: Store BOTH text and images in Chroma (NEW - MULTIMODAL)
        print("\n💾 Storing in vector database...")
        collection = init_chroma()
        
        # Store text
        if embedded_text_items:
            store_chunks(collection, embedded_text_items)
            print(f"✅ Stored {len(embedded_text_items)} text chunks")
        
        # Store images
        if embedded_images:
            try:
                store_chunks(collection, embedded_images)
                print(f"✅ Stored {len(embedded_images)} images")
            except Exception as e:
                print(f"⚠️  Error storing images: {e}")
        
        # STEP 7: Test retrieval (NEW - MULTIMODAL TEST)
        print("\n🧪 Testing multimodal retrieval...")
        
        # Test text retrieval
        test_query(collection)
        
        # Test image retrieval
        print("🔍 Testing image retrieval...")
        try:
            result = collection.query(query_texts=["figure diagram chart"], n_results=3)
            if result and result.get("documents"):
                images_found = [doc for doc in result["documents"][0] if "Image:" in doc]
                print(f"✅ Image retrieval test: Found {len(images_found)} images")
        except Exception as e:
            print(f"⚠️  Image retrieval test failed: {e}")
        
        print("\n🎯 TRUE MULTIMODAL Ingestion Complete ✅")
        print(f"📊 Summary:")
        print(f"   Text chunks: {len(embedded_text_items)}")
        print(f"   Images: {len(embedded_images)}")
        print(f"   Total vectors: {len(embedded_text_items) + len(embedded_images)}")
        
    except Exception as e:
        print(f"❌ Error during PDF processing: {str(e)}")
        raise