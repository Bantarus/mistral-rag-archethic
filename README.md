# Archethic Documentation RAG System

This project implements a Retrieval-Augmented Generation (RAG) system for the Archethic documentation using LangChain and FAISS.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your environment:
   - Create either `.env.local` (recommended) or `.env` file
   - Add your Mistral API key to the file:
     ```
     MISTRAL_API_KEY=your_api_key_here
     ```
   Note: `.env.local` takes precedence over `.env` if both exist

3. Place your documentation in the `docs/archethic-docs/docs` directory or update the path in `rag_system.py`

## Usage

Run the system:
```bash
python rag_system.py
```

The script will:
1. Load and process your documentation
2. Create a FAISS vector store
3. Save the index for future use
4. Run an example query

To use the RAG system in your code:

```python
from rag_system import create_rag_system, setup_qa_chain, query_docs

# Initialize the system
vector_store = create_rag_system("path/to/docs", api_key)
qa_chain = setup_qa_chain(vector_store, api_key)

# Query the documentation
answer = query_docs(qa_chain, "What is Archethic?")
print(answer)
```

## Features

- Recursive document loading from directory
- Efficient text chunking with overlap
- FAISS vector store for fast similarity search
- Mistral AI embeddings and chat model
- Persistent vector store index
- Clear and focused responses based on context 