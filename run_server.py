#!/usr/bin/env python3
"""
Simple test script to check if backend can start
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    print("Testing backend startup...")
    
    try:
        print("1. Importing FastAPI...")
        from fastapi import FastAPI
        print("   ✅ FastAPI imported")
        
        print("2. Importing vectordb...")
        from vectordb.chroma_client import init_chroma
        print("   ✅ Chroma client imported")
        
        print("3. Importing ingestion...")
        from pipeline.ingest_pipeline import run_ingestion
        print("   ✅ Ingestion pipeline imported")
        
        print("\n4. Starting backend...")
        import uvicorn
        from api import app
        
        print("✅ All modules loaded successfully!")
        print("\n🚀 Starting server on http://0.0.0.0:8001")
        uvicorn.run(app, host="0.0.0.0", port=8001)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
