# pipeline/ingest_pipeline.py
# Delegates to multimodal_ingestion.py for complete pipeline

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from pipeline.multimodal_ingestion import run_ingestion, get_pipeline
except ImportError:
    from backend.pipeline.multimodal_ingestion import run_ingestion, get_pipeline

__all__ = ["run_ingestion", "get_pipeline"]