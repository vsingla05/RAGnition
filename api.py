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
except ImportError:
    from pipeline.ingest_pipeline import run_ingestion
    from vectordb.chroma_client import init_chroma
    from retrieval.retrieval_pipeline import run_retrieval

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)