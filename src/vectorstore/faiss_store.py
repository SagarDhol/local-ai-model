import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import pickle
import json
from colorama import Fore, Style

class FAISSStore:
    def __init__(self, vector_dim: int, index_path: Optional[Path] = None):
        self.vector_dim = vector_dim
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(vector_dim)
        self.documents = []
        
        if index_path and index_path.exists():
            self.load(index_path)
    
    def add_embeddings(self, embeddings: List[List[float]], documents: List[Dict[str, Any]]):
        """Add document embeddings to the index with metadata."""
        if not embeddings:
            return
            
        vectors = np.array(embeddings).astype('float32')
        self.index.add(vectors)
        
        # Store document metadata
        for doc in documents:
            # Store a serializable version of the document
            self.documents.append({
                'text': doc.get('text', ''),
                'source': doc.get('source', 'unknown'),
                'chunk_id': doc.get('chunk_id', -1),
                'total_chunks': doc.get('total_chunks', 0)
            })
        
        print(f"{Fore.GREEN}✓ Added {len(documents)} documents to the vector store{Style.RESET_ALL}")
    
    def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents with metadata and scoring."""
        if len(self.documents) == 0:
            print(f"{Fore.YELLOW}Warning: No documents in the vector store{Style.RESET_ALL}")
            return []
            
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(distances[0][i])))
        
        # Sort by score (distance) - lower is better
        results.sort(key=lambda x: x[1])
        
        print(f"{Fore.GREEN}✓ Found {len(results)} relevant documents{Style.RESET_ALL}")
        return results
    
    def save(self, path: Path):
        """Save the index and documents with error handling."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(path))
            
            # Save documents with JSON serialization for better compatibility
            serializable_docs = []
            for doc in self.documents:
                if isinstance(doc, dict):
                    serializable_docs.append(doc)
                else:
                    # Fallback for non-dict documents
                    serializable_docs.append({'text': str(doc)})
            
            with open(f"{path}.pkl", "wb") as f:
                pickle.dump(serializable_docs, f)
                
            print(f"{Fore.GREEN}✓ Saved vector store to {path}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Error saving vector store: {str(e)}{Style.RESET_ALL}")
            raise
    
    def load(self, path: Path):
        """Load the index and documents with error handling."""
        try:
            if not path.exists():
                print(f"{Fore.YELLOW}No existing vector store found at {path}, creating a new one{Style.RESET_ALL}")
                return
                
            print(f"{Fore.CYAN}Loading vector store from {path}...{Style.RESET_ALL}")
            self.index = faiss.read_index(str(path))
            
            try:
                with open(f"{path}.pkl", "rb") as f:
                    self.documents = pickle.load(f)
                print(f"{Fore.GREEN}✓ Loaded {len(self.documents)} documents{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.YELLOW}Warning: Could not load document metadata: {str(e)}{Style.RESET_ALL}")
                self.documents = []
                
        except Exception as e:
            print(f"{Fore.RED}Error loading vector store: {str(e)}{Style.RESET_ALL}")
            raise
