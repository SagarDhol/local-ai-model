# Local RAG Demo with LLaMA 3

A complete local RAG (Retrieval-Augmented Generation) demo using LLaMA 3, nomic-embed-text, and FAISS.

## Setup

1. Install Ollama and pull the required models:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Index documents:
   ```bash
   python main.py index --file data/documents.txt
   ```

2. Query the system:
   ```bash
   python main.py query "What is LLaMA 3?"
   ```

## Project Structure

- `config/`: Configuration settings
- `data/`: Input documents
- `src/`: Source code
  - `embedding/`: Text embedding functionality
  - `llm/`: LLaMA 3 client
  - `vectorstore/`: FAISS vector store operations
  - `utils/`: Utility functions
- `main.py`: Main entry point

## How It Works

1. **Embedding Generation**: Text is converted to vector embeddings using nomic-embed-text
2. **Vector Storage**: Embeddings are stored in a FAISS index for efficient similarity search
3. **Retrieval**: Queries are converted to embeddings and similar documents are retrieved
4. **Generation**: Retrieved context is combined with the query and sent to LLaMA 3 for response generation
