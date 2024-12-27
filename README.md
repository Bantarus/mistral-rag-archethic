# Archethic Documentation RAG System

This project implements a Retrieval-Augmented Generation (RAG) system for the Archethic documentation using Mistralai, LangChain and FAISS.

## Features

Current Features:
- Recursive document loading from directory
- Efficient text chunking with overlap
- FAISS vector store for fast similarity search
- Mistral AI embeddings and chat model
- Persistent vector store index
- Document registry for incremental updates
- Rate-limited batch processing
- UTF-8 encoding support
- Clear and focused responses based on context

Limitations:
- Currently implements a simple Question/Answer system without conversation memory
- No support yet for follow-up questions or context from previous interactions
  > Note: Conversation memory could be implemented by modifying the prompt template to include previous interactions, while respecting the model's context length limits. This is a planned future enhancement.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your environment:
   - Create either `.env.local` (recommended) or `.env` file
   - Add the following environment variables:
     ```
     # Required for embeddings via Mistral API
     MISTRAL_API_KEY=your_api_key_here
     
     # Required for local model inference
     MISTRAL_MODEL_PATH=path/to/your/mistral/model
     ```
   Note: 
   - `.env.local` takes precedence over `.env` if both exist
   - The MISTRAL_API_KEY is used only for generating embeddings via the Mistral API service
   - The MISTRAL_MODEL_PATH should point to your downloaded model directory
   - The free tier of Mistral API has rate limits, which is why we implement lazy loading and rate limiting for embeddings

   ⚠️ **Warning**: The system currently sets `KMP_DUPLICATE_LIB_OK=TRUE` to handle multiple OpenMP instances. For optimal performance and consistency, it's recommended to run only one OpenMP  instance. This is a temporary workaround that may impact performance.

3. Download a compatible model:
   Available models from [mistral-inference](https://github.com/mistralai/mistral-inference):
   - Mistral 7B Instruct v0.3
   - Mixtral 8x7B Instruct v0.1
   - Mixtral 8x22B Instruct v0.3
   - Mistral 7B Base v0.3
   - Mixtral 8x22B v0.3
   - Codestral 22B v0.1
   - Mathstral 7B v0.1
   - Codestral-Mamba 7B v0.1
   - Mistral Nemo Base/Instruct
   - Mistral Large 2

4. Initialize the document registry:
```bash
python create_registry.py
```
This creates a registry of all markdown files in the docs directory, tracking their content hashes for incremental updates.

## Rate Limiting and Lazy Loading

The system implements two key features to work within Mistral API's free tier limitations:

1. **Rate Limiting**: 
   - Limits embedding requests to respect the API's rate limit
   - Configurable batch size (default: 1) and delay between batches (default: 2 seconds, could work with 1 sec, not tested)
   - Prevents API throttling and ensures reliable processing

2. **Lazy Loading**:
   - Documents are loaded and processed only when needed
   - Reduces memory usage for large documentation sets
   - Enables processing of documents in smaller batches
   - Tracks processed documents to avoid unnecessary API calls

These features ensure reliable operation within the free tier limits while maintaining system functionality.

## Usage

Run the system:
```bash
python rag_system.py
```

The script will:
1. Load the document registry
2. Process only new or modified documents
3. Create or update the FAISS vector store
4. Run an interactive query session

### Directory Structure

Place your documentation in the `docs` directory. The system will:
- Recursively scan all subdirectories
- Process all markdown (`.md`) files
- Maintain directory structure in metadata
- Track file changes for incremental updates

### Using in Code

```python
from rag_system import LazyRAGSystem, setup_qa_chain, query_docs

# Initialize the system
embeddings = MistralAIEmbeddings(mistral_api_key=your_api_key)
rag_system = LazyRAGSystem(embeddings)

# Load documents with UTF-8 encoding
loader = DirectoryLoader(
    "docs",
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'},
    show_progress=True
)

# Process documents with rate limiting
vector_store = rag_system.process_documents(
    list(loader.lazy_load()),
    batch_size=1,
    rate_limit_delay=2.0
)

# Setup QA chain
qa_chain = setup_qa_chain(vector_store, local_model)

# Query the documentation
answer = query_docs(qa_chain, "What is Archethic?")
print(answer)
```

## Document Registry

The system maintains a registry (`faiss_index_docs_registry.json`) to track processed documents:
- Uses SHA-256 hashing for content verification
- Stores modification timestamps
- Enables incremental updates
- Prevents reprocessing unchanged files

## Rate Limiting

The system includes rate limiting for API calls:
- Configurable batch size (default: 1)
- Adjustable delay between batches (default: 2 seconds)
- Progress tracking for batch processing

## File Handling

- Recursive document loading from directory
- Efficient text chunking with overlap
- FAISS vector store for fast similarity search
- Mistral AI embeddings and chat model
- Persistent vector store index
- Clear and focused responses based on context 
- UTF-8 encoding support for international characters
- Windows and Unix path compatibility
- Recursive directory traversal
- Metadata preservation for document relationships 