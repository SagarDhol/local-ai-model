from typing import List
import re

def split_into_chunks(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """Split text into overlapping chunks."""
    words = re.split(r'\s+', text)
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    
    return chunks
