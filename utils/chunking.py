# utils/chunking.py

from typing import List
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def is_image_caption(line: str) -> bool:
    l = line.lower().strip()

    return (
        l.startswith("figure")
        or l.startswith("fig.")
        or "figure " in l
        or "fig." in l
    )