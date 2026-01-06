import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import argparse
from dotenv import load_dotenv
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

from config import settings
from src.embedding.nomic_embedder import NomicEmbedder
from src.vectorstore.faiss_store import FAISSStore
from src.llm.llama3_client import Llama3Client
from src.utils.file_utils import split_into_chunks

# Color definitions
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class RAGDemo:
    def __init__(self):
        print(f"{Colors.CYAN}{'='*50}\nInitializing RAG System\n{'='*50}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Loading embedding model: {settings.EMBEDDING_MODEL}{Colors.ENDC}")
        self.embedder = NomicEmbedder(settings.EMBEDDING_MODEL)
        print(f"{Colors.YELLOW}Loading LLM model: {settings.LLM_MODEL}{Colors.ENDC}")
        self.llm = Llama3Client(settings.LLM_MODEL)
        print(f"{Colors.YELLOW}Initializing vector store...{Colors.ENDC}")
        self.vector_store = FAISSStore(
            vector_dim=768,  # nomic-embed-text uses 768 dimensions
            index_path=settings.VECTOR_STORE_PATH
        )
        print(f"{Colors.GREEN}RAG System initialized successfully!\n{'-'*50}{Colors.ENDC}\n")
    
    def load_documents(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load and split documents into chunks with metadata."""
        print(f"{Colors.BLUE}Loading documents from: {file_path}{Colors.ENDC}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into chunks
        chunks = split_into_chunks(
            text, 
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP
        )
        
        # Add metadata to chunks
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            chunk_docs.append({
                'text': chunk,
                'source': str(file_path),
                'chunk_id': i,
                'total_chunks': len(chunks)
            })
        
        print(f"{Colors.GREEN}✓ Split into {len(chunk_docs)} chunks{Colors.ENDC}")
        return chunk_docs
    
    def index_documents(self, document_path: Path):
        """Index documents in the vector store with detailed logging."""
        print(f"\n{Colors.HEADER}=== Document Indexing Process ==={Colors.ENDC}")
        
        # Load and split documents
        chunks = self.load_documents(document_path)
        
        # Generate embeddings
        print(f"\n{Colors.CYAN}Generating embeddings for {len(chunks)} chunks...{Colors.ENDC}")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.get_embeddings(texts)
        
        # Show sample embedding
        if embeddings:
            sample_embedding = embeddings[0]
            print(f"{Colors.YELLOW}Sample embedding (first 5 dims): {sample_embedding[:5]}...{Colors.ENDC}")
        
        # Add to vector store
        print(f"\n{Colors.CYAN}Indexing documents in vector store...{Colors.ENDC}")
        self.vector_store.add_embeddings(embeddings, chunks)
        self.vector_store.save(settings.VECTOR_STORE_PATH)
        
        print(f"\n{Colors.GREEN}✓ Successfully indexed {len(chunks)} document chunks in {settings.VECTOR_STORE_PATH}{Colors.ENDC}")
        print(f"{'-'*50}\n")
    
    def query(self, question: str, top_k: int = 3) -> str:
        """Query the RAG system with detailed logging."""
        print(f"\n{Colors.HEADER}=== Query Processing ==={Colors.ENDC}")
        
        # Get query embedding
        print(f"{Colors.CYAN}Generating embedding for query...{Colors.ENDC}")
        query_embedding = self.embedder.get_embedding(question)
        print(f"{Colors.YELLOW}Query embedding (first 5 dims): {query_embedding[:5]}...{Colors.ENDC}")
        
        # Retrieve relevant documents
        print(f"\n{Colors.CYAN}Searching for relevant documents...{Colors.ENDC}")
        results = self.vector_store.similarity_search(query_embedding, k=top_k)
        
        # Show retrieved chunks
        print(f"\n{Colors.GREEN}✓ Retrieved {len(results)} relevant chunks:{Colors.ENDC}")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n{Colors.YELLOW}--- Chunk {i} (Score: {score:.4f}) ---{Colors.ENDC}")
            print(f"{doc['text'][:200]}...")
        
        # Format context
        context = "\n\n".join([doc['text'] for doc, _ in results])
        
        # Generate response
        print(f"\n{Colors.CYAN}Generating response using LLaMA 3...{Colors.ENDC}")
        response = self.llm.generate(
            prompt=question,
            context=context
        )
        
        # Format and return response
        print(f"\n{Colors.GREEN}{'='*50}")
        print(f"{Colors.GREEN}{'RESPONSE':^50}{Colors.ENDC}")
        print(f"{Colors.GREEN}{'='*50}{Colors.ENDC}")
        print(f"{Colors.BLUE}{response}{Colors.ENDC}")
        print(f"{Colors.GREEN}{'='*50}{Colors.ENDC}")
        
        return response

def interactive_mode():
    """Run in interactive mode with prompts for user input."""
    print(f"\n{Colors.HEADER}=== Interactive RAG System ==={Colors.ENDC}")
    print(f"{Colors.CYAN}Choose an action:{Colors.ENDC}")
    print(f"1. Index documents")
    print(f"2. Query the system")
    print(f"3. Exit")
    
    while True:
        try:
            choice = input(f"\n{Colors.YELLOW}Enter your choice (1-3): {Colors.ENDC}").strip()
            
            if choice == '1':
                # Index documents
                file_path = input(f"\n{Colors.CYAN}Enter the path to the document to index (or press Enter to use 'data/documents.txt'): {Colors.ENDC}")
                file_path = Path(file_path) if file_path.strip() else Path('data/documents.txt')
                
                # Create the file if it doesn't exist
                if not file_path.exists():
                    create_file = input(f"{Colors.YELLOW}File not found. Create it? (y/n): {Colors.ENDC}").lower()
                    if create_file == 'y':
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(file_path, 'w') as f:
                            f.write("""The LLaMA (Large Language Model Meta AI) is a collection of large language models developed by Meta AI. 
The models range in size from 7 billion to 65 billion parameters and are designed to be more efficient than previous models.

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.""")
                        print(f"{Colors.GREEN}Created default document at {file_path}{Colors.ENDC}")
                    else:
                        continue
                
                rag = RAGDemo()
                rag.index_documents(file_path)
                
            elif choice == '2':
                # Query the system
                question = input(f"\n{Colors.CYAN}Enter your question (or press Enter for a default question): {Colors.ENDC}")
                if not question.strip():
                    question = "What is LLaMA 3?"
                    print(f"{Colors.YELLOW}Using default question: {question}{Colors.ENDC}")
                
                top_k = input(f"{Colors.CYAN}Number of documents to retrieve (default: 3): {Colors.ENDC}")
                top_k = int(top_k) if top_k.strip() and top_k.isdigit() else 3
                
                rag = RAGDemo()
                rag.query(question, top_k)
                
            elif choice == '3':
                print(f"{Colors.GREEN}Exiting...{Colors.ENDC}")
                break
                
            else:
                print(f"{Colors.RED}Invalid choice. Please enter a number between 1 and 3.{Colors.ENDC}")
                
        except KeyboardInterrupt:
            print(f"\n{Colors.RED}Operation cancelled by user.{Colors.ENDC}")
            break
        except Exception as e:
            print(f"{Colors.RED}Error: {str(e)}{Colors.ENDC}")
            continue

def main():
    load_dotenv()
    
    # If no arguments provided, run in interactive mode
    if len(sys.argv) == 1:
        interactive_mode()
        return
    
    # Otherwise, use command-line arguments
    parser = argparse.ArgumentParser(
        description=f"{Colors.HEADER}Local RAG Demo with LLaMA 3{Colors.ENDC}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"{Colors.CYAN}Example usage:\n"
               "  Interactive mode: python main.py\n"
               "  Index documents: python main.py index --file data/documents.txt\n"
               "  Query the system: python main.py query \"What is LLaMA 3?\"{Colors.ENDC}"
    )
    
    subparsers = parser.add_subparsers(dest='command', required=False)
    
    # Index command
    index_parser = subparsers.add_parser(
        'index', 
        help='Index documents',
        description=f"{Colors.BLUE}Index documents for RAG system{Colors.ENDC}"
    )
    index_parser.add_argument(
        '--file', 
        type=Path, 
        default=Path('data/documents.txt'),
        help='Path to the text file to index (default: data/documents.txt)'
    )
    
    # Query command
    query_parser = subparsers.add_parser(
        'query', 
        help='Query the RAG system',
        description=f"{Colors.BLUE}Query the RAG system{Colors.ENDC}"
    )
    query_parser.add_argument(
        'question', 
        type=str, 
        nargs='?',
        default=None,
        help='The question to ask the system (can be provided interactively)'
    )
    query_parser.add_argument(
        '--top-k', 
        type=int, 
        default=3,
        help='Number of document chunks to retrieve (default: 3)'
    )
    
    args = parser.parse_args()
    
    # If no command is provided, run in interactive mode
    if not args.command:
        interactive_mode()
        return
    
    rag = RAGDemo()
    
    if args.command == 'index':
        rag.index_documents(args.file)
    elif args.command == 'query':
        if not args.question:
            args.question = input(f"{Colors.CYAN}Enter your question (or press Enter to use a default question): {Colors.ENDC}")
            if not args.question.strip():
                args.question = "What is LLaMA 3?"
                print(f"{Colors.YELLOW}Using default question: {args.question}{Colors.ENDC}")
        rag.query(args.question, args.top_k)

if __name__ == "__main__":
    main()
