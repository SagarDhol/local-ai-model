import ollama
from typing import List
from tqdm import tqdm

class NomicEmbedder:
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        response = ollama.embeddings(model=self.model_name, prompt=text)
        return response["embedding"]
    
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for multiple texts with progress bar."""
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = [self.get_embedding(text) for text in batch]
            embeddings.extend(batch_embeddings)
        return embeddings