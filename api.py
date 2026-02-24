"""
FastAPI backend for RAG system with PDF upload and Q&A
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from backend.pipeline.ingest_pipeline import run_ingestion
    from backend.vectordb.chroma_client import init_chroma
    from backend.retrieval.retrieval_pipeline import run_retrieval
    from backend.retrieval.multimodal_pipeline import MultimodalRetrievalPipeline
except ImportError:
    from pipeline.ingest_pipeline import run_ingestion
    from vectordb.chroma_client import init_chroma
    from retrieval.retrieval_pipeline import run_retrieval
    from retrieval.multimodal_pipeline import MultimodalRetrievalPipeline

# Initialize FastAPI
app = FastAPI(title="RAG System API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (configure for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global collection object
collection = None
multimodal_pipeline = None


# Pydantic models
class QuestionRequest(BaseModel):
    question: str


class UploadResponse(BaseModel):
    status: str
    filename: str
    message: str


class QuestionResponse(BaseModel):
    response: str
    source_chunks: list = []
    confidence: float = 0.0


# Initialize collection on startup
@app.on_event("startup")
async def startup_event():
    global collection
    print("🚀 Initializing Chroma database...")
    collection = init_chroma()
    print("✅ Database initialized!")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RAG System API"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF file
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Validate file size (max 500MB)
        file_size = 0
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        if file_size > 500 * 1024 * 1024:  # 500MB limit
            raise HTTPException(status_code=400, detail="File size exceeds 500MB limit")
        
        # Reset file pointer and save
        await file.seek(0)
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"\n📄 Uploading: {file.filename} ({file_size / 1024 / 1024:.2f}MB)")
        
        # Run ingestion
        print(f"� Processing PDF...")
        run_ingestion(file_path)
        print(f"✅ Successfully ingested {file.filename}\n")
        
        return UploadResponse(
            status="success",
            filename=file.filename,
            message=f"PDF '{file.filename}' uploaded and ingested successfully!"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error during ingestion: {error_msg}\n")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {error_msg}")


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the ingested documents
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if collection is None:
            raise HTTPException(status_code=503, detail="Database not initialized")
        
        print(f"🔎 Processing question: {request.question}")
        
        # Run retrieval pipeline
        result = run_retrieval(collection, request.question)
        
        # Extract response (handle different return formats)
        if isinstance(result, dict):
            response_text = result.get("response", "No answer found")
            source_chunks = result.get("source_chunks", [])
            confidence = result.get("confidence", 0.0)
        else:
            response_text = str(result) if result else "No answer found"
            source_chunks = []
            confidence = 0.0
        
        return QuestionResponse(
            response=response_text,
            source_chunks=source_chunks,
            confidence=confidence
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during retrieval: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")


@app.get("/documents")
async def get_documents():
    """
    Get list of ingested documents
    """
    try:
        if collection is None:
            return {"documents": []}
        
        # Get all documents from collection
        documents = collection.get()
        doc_list = []
        
        if documents and "documents" in documents:
            for i, doc in enumerate(documents["documents"]):
                doc_list.append({
                    "id": documents["ids"][i] if "ids" in documents else str(i),
                    "content": doc[:200] + "..." if len(doc) > 200 else doc
                })
        
        return {"documents": doc_list, "count": len(doc_list)}
    
    except Exception as e:
        print(f"❌ Error fetching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")


# ============================================================================
# MULTIMODAL ENDPOINTS - Image, Table, and Cross-modal Retrieval
# ============================================================================

class MultimodalQuestionResponse(BaseModel):
    response: str
    text_sources: list = []
    image_sources: list = []
    table_sources: list = []
    confidence: float = 0.0


@app.post("/ask-multimodal", response_model=MultimodalQuestionResponse)
async def ask_multimodal_question(request: QuestionRequest):
    """
    Ask a question with multimodal retrieval
    Finds answers in text, images, tables, and diagrams
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if collection is None:
            raise HTTPException(status_code=503, detail="Database not initialized")
        
        print(f"🔎 Processing multimodal question: {request.question}")
        
        # Run multimodal pipeline if available
        if multimodal_pipeline:
            try:
                result = multimodal_pipeline.run_full_pipeline(request.question)
                
                return MultimodalQuestionResponse(
                    response=result.get("answer", "No answer found"),
                    text_sources=[s for s in result.get("sources", []) if s.get("type") == "text"],
                    image_sources=[s for s in result.get("sources", []) if s.get("type") == "image"],
                    table_sources=[s for s in result.get("sources", []) if s.get("type") == "table"],
                    confidence=0.85
                )
            except Exception as e:
                print(f"⚠️  Multimodal pipeline error: {e}, falling back to text retrieval")
        
        # Fallback to regular retrieval if multimodal not available
        result = run_retrieval(collection, request.question)
        
        if isinstance(result, dict):
            response_text = result.get("response", "No answer found")
            source_chunks = result.get("source_chunks", [])
        else:
            response_text = str(result) if result else "No answer found"
            source_chunks = []
        
        return MultimodalQuestionResponse(
            response=response_text,
            text_sources=[{"type": "text", "content": chunk} for chunk in source_chunks],
            confidence=0.75
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in multimodal retrieval: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Multimodal question processing failed: {str(e)}")


@app.get("/retrieve-images")
async def retrieve_images(query: str):
    """
    Retrieve images most relevant to a query
    Uses CLIP embeddings for cross-modal search
    """
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if not multimodal_pipeline:
            return {"images": [], "message": "Multimodal pipeline not initialized"}
        
        print(f"🖼️  Retrieving images for: {query}")
        
        retrieved = multimodal_pipeline._retrieve_images(query, top_k=5)
        
        return {
            "query": query,
            "images": retrieved,
            "count": len(retrieved)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error retrieving images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image retrieval failed: {str(e)}")


@app.get("/retrieve-tables")
async def retrieve_tables(query: str):
    """
    Retrieve tables most relevant to a query
    """
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if not multimodal_pipeline:
            return {"tables": [], "message": "Multimodal pipeline not initialized"}
        
        print(f"📊 Retrieving tables for: {query}")
        
        retrieved = multimodal_pipeline._retrieve_tables(query)
        
        return {
            "query": query,
            "tables": retrieved,
            "count": len(retrieved)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error retrieving tables: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Table retrieval failed: {str(e)}")


@app.get("/query-analysis")
async def analyze_query(query: str):
    """
    Analyze query to determine what type of information to retrieve
    Returns: primary modality, secondary modalities, confidence
    """
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        from retrieval.multimodal_router import MultimodalQueryRouter
        router = MultimodalQueryRouter()
        analysis = router.analyze_query(query)
        
        return {
            "query": query,
            "primary_modality": analysis["primary_modality"].value,
            "secondary_modalities": [m.value for m in analysis["secondary_modalities"]],
            "confidence": analysis["confidence"],
            "recommendation": analysis["recommendation"],
            "keywords_found": analysis["keywords_found"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error analyzing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)