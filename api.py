"""
FastAPI backend for Engineering Document Intelligence System
Multimodal RAG: Text + Images + Tables per uploaded PDF
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import base64

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from pipeline.multimodal_ingestion import run_ingestion, get_pipeline
    from vectordb.chroma_client import init_chroma
    from retrieval.multimodal_pipeline import run_multimodal_rag
except ImportError:
    from backend.pipeline.multimodal_ingestion import run_ingestion, get_pipeline
    from backend.vectordb.chroma_client import init_chroma
    from backend.retrieval.multimodal_pipeline import run_multimodal_rag

# Initialize FastAPI
app = FastAPI(
    title="Engineering Document Intelligence System",
    description="Multimodal RAG for engineering manuals: text, images, tables",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
IMAGES_DIR = BASE_DIR / "extracted_images"
UPLOAD_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

# Serve extracted images as static files
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

# Global state
collection = None

# Document registry: maps doc_id -> document info
# This is the source of truth for "which document is currently active"
DOCUMENT_REGISTRY: dict = {}

# Track the MOST RECENTLY uploaded document (for single-doc mode)
CURRENT_DOC_ID: str = None


# ============================================================================
# Pydantic models
# ============================================================================

class QuestionRequest(BaseModel):
    question: str
    doc_id: str = None  # Optional: if provided, search only this doc


class UploadResponse(BaseModel):
    status: str
    filename: str
    message: str
    doc_id: str = ""


class QuestionResponse(BaseModel):
    response: str
    source_chunks: list = []
    confidence: float = 0.0


class MultimodalQuestionResponse(BaseModel):
    response: str
    text_sources: list = []
    image_sources: list = []
    table_sources: list = []
    confidence: float = 0.0
    doc_id: str = ""
    modalities: dict = {}


# ============================================================================
# Startup
# ============================================================================

@app.on_event("startup")
async def startup_event():
    global collection
    print("🚀 Initializing Engineering Document Intelligence System...")
    collection = init_chroma()
    print("✅ Vector database initialized!")


# ============================================================================
# Health & Info
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Engineering Document Intelligence System",
        "version": "2.0.0",
        "current_doc": CURRENT_DOC_ID
    }


@app.get("/documents")
async def get_documents():
    """Get list of all uploaded/ingested documents"""
    try:
        docs = []
        for doc_id, info in DOCUMENT_REGISTRY.items():
            docs.append({
                "doc_id": doc_id,
                "name": info.get("name", ""),
                "filename": info.get("filename", ""),
                "timestamp": info.get("timestamp", ""),
                "text_chunks": info.get("text_chunks", 0),
                "images": info.get("images", 0),
                "tables": info.get("tables", 0),
                "total_vectors": info.get("total_vectors", 0),
                "is_current": doc_id == CURRENT_DOC_ID
            })
        # Most recent first
        docs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return {
            "documents": docs,
            "count": len(docs),
            "current_doc_id": CURRENT_DOC_ID
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")


@app.get("/current-document")
async def get_current_document():
    """Get the most recently uploaded document info"""
    if not CURRENT_DOC_ID:
        return {"doc_id": None, "message": "No document uploaded yet"}
    info = DOCUMENT_REGISTRY.get(CURRENT_DOC_ID, {})
    return {
        "doc_id": CURRENT_DOC_ID,
        **info
    }


# ============================================================================
# PDF Upload
# ============================================================================

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF file.
    After upload, all /ask-multimodal calls will use THIS document
    unless a specific doc_id is provided.
    """
    global CURRENT_DOC_ID

    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Read file content
        file_content = await file.read()
        file_size = len(file_content)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="File is empty")

        if file_size > 500 * 1024 * 1024:  # 500MB limit
            raise HTTPException(status_code=400, detail="File size exceeds 500MB limit")

        # Save file
        safe_filename = Path(file.filename).name  # Remove any path components
        file_path = UPLOAD_DIR / safe_filename
        with open(str(file_path), "wb") as buffer:
            buffer.write(file_content)

        print(f"\n📄 Uploading: {file.filename} ({file_size / 1024 / 1024:.2f}MB)")

        # Run multimodal ingestion (returns doc_id)
        print(f"🔄 Processing PDF with multimodal pipeline...")
        pipeline = get_pipeline()
        doc_id = pipeline.ingest_document(str(file_path))

        # Get registration info from the pipeline
        registry = pipeline.get_document_registry()
        doc_info = registry.get(doc_id, {})

        # Update global registry
        DOCUMENT_REGISTRY[doc_id] = {
            **doc_info,
            "filename": safe_filename,
            "timestamp": datetime.now().isoformat()
        }

        # Set as current document
        CURRENT_DOC_ID = doc_id

        print(f"✅ Successfully ingested: {file.filename} (doc_id: {doc_id})\n")

        return UploadResponse(
            status="success",
            filename=file.filename,
            message=f"PDF '{file.filename}' uploaded and processed successfully! "
                    f"Extracted {doc_info.get('text_chunks', 0)} text chunks, "
                    f"{doc_info.get('images', 0)} images, "
                    f"{doc_info.get('tables', 0)} tables.",
            doc_id=doc_id
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error during ingestion: {error_msg}\n")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {error_msg}")


# ============================================================================
# Q&A Endpoints
# ============================================================================

@app.post("/ask-multimodal", response_model=MultimodalQuestionResponse)
async def ask_multimodal_question(request: QuestionRequest):
    """
    Ask a question with multimodal retrieval.
    
    - If doc_id is provided -> search only that document
    - If no doc_id -> search the most recently uploaded document (CURRENT_DOC_ID)
    - This ensures answers come from the CURRENT PDF, not previous ones
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Determine which document to search
        target_doc_id = request.doc_id or CURRENT_DOC_ID

        if not target_doc_id:
            raise HTTPException(
                status_code=400,
                detail="No document uploaded yet. Please upload a PDF first."
            )

        doc_info = DOCUMENT_REGISTRY.get(target_doc_id, {})
        doc_name = doc_info.get("name") or doc_info.get("filename", "uploaded document")

        print(f"\n🔎 Question: {request.question}")
        print(f"📋 Searching document: {doc_name} (doc_id: {target_doc_id})")

        # Run multimodal RAG with strict document filtering
        result = run_multimodal_rag(
            query=request.question,
            doc_id=target_doc_id,
            top_k=5
        )

        if result.get("error"):
            print(f"❌ Pipeline error: {result['error']}")
            raise HTTPException(
                status_code=500,
                detail=f"Question processing failed: {result['error']}"
            )

        return MultimodalQuestionResponse(
            response=result.get("answer", "No answer found"),
            text_sources=result.get("sources", []),
            image_sources=result.get("images_referenced", []),
            table_sources=[],  # Tables are included in sources
            confidence=result.get("confidence", 0.0),
            doc_id=target_doc_id,
            modalities=result.get("modalities", {})
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in multimodal retrieval: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Multimodal question processing failed: {str(e)}"
        )


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Legacy text-only Q&A endpoint.
    Delegates to /ask-multimodal for consistency.
    """
    response = await ask_multimodal_question(request)
    return {
        "response": response.response,
        "source_chunks": response.text_sources,
        "confidence": response.confidence
    }


# ============================================================================
# Image serving
# ============================================================================

@app.get("/image/{image_filename}")
async def get_image(image_filename: str):
    """Serve extracted images to frontend"""
    try:
        # Security: prevent directory traversal
        if ".." in image_filename or "/" in image_filename:
            raise HTTPException(status_code=400, detail="Invalid image filename")

        image_path = IMAGES_DIR / image_filename

        if not image_path.exists():
            print(f"⚠️  Image not found: {image_path}")
            raise HTTPException(status_code=404, detail=f"Image not found: {image_filename}")

        # Determine content type
        ext = image_path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        media_type = media_types.get(ext, "image/png")

        return FileResponse(str(image_path), media_type=media_type)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")


@app.get("/image-base64/{image_filename}")
async def get_image_base64(image_filename: str):
    """Get image as base64 encoded string"""
    try:
        if ".." in image_filename or "/" in image_filename:
            raise HTTPException(status_code=400, detail="Invalid image filename")

        image_path = IMAGES_DIR / image_filename

        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {image_filename}")

        with open(str(image_path), "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode("utf-8")

        ext = image_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        mime_type = mime_types.get(ext, "image/png")

        return {
            "filename": image_filename,
            "base64": f"data:{mime_type};base64,{image_data}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encoding image: {str(e)}")


@app.get("/document-images/{doc_id}")
async def get_document_images(doc_id: str):
    """Get all images extracted from a specific document"""
    try:
        doc_info = DOCUMENT_REGISTRY.get(doc_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

        # List images with doc_name prefix
        doc_name = doc_info.get("name", "")
        images = []
        for img_path in IMAGES_DIR.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                images.append({
                    "filename": img_path.name,
                    "url": f"/image/{img_path.name}",
                    "size": img_path.stat().st_size
                })

        return {"doc_id": doc_id, "images": images, "count": len(images)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)