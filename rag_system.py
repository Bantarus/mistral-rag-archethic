import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from typing import List, Union, Dict, Generator
from datetime import datetime
import json
from pathlib import Path
from langchain_core.documents import Document
import time  # Add this import at the top
import hashlib  # Add this import at the top

# Load environment variables, prioritizing .env.local
load_dotenv(dotenv_path='.env.local')  # Try .env.local first
if not os.getenv("MISTRAL_API_KEY"):  # If API key not found, try .env as fallback
    load_dotenv(dotenv_path='.env')

# Replace with environment variable
MISTRAL_MODEL_PATH = os.getenv("MISTRAL_MODEL_PATH")
if not MISTRAL_MODEL_PATH:
    raise ValueError("MISTRAL_MODEL_PATH environment variable is not set")

# Initialize tokenizer and model
tokenizer = MistralTokenizer.from_file(os.path.join(MISTRAL_MODEL_PATH, "tokenizer.model.v3"))
model = Transformer.from_folder(MISTRAL_MODEL_PATH)

class LocalMistralChat:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.35):
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
        out_tokens, _ = generate(
            [tokens], 
            self.model, 
            max_tokens=max_tokens, 
            temperature=temperature, 
            eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id
        )
        return self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

class LazyRAGSystem:
    def __init__(self, embeddings_model, index_name: str = "faiss_index_docs"):
        self.embeddings = embeddings_model
        self.index_name = index_name
        self.document_registry = {}
        # Print the absolute path of the registry file
        self.registry_path = os.path.abspath(f"{index_name}_registry.json")
        print(f"Registry will be saved to: {self.registry_path}")
        self.base_path = Path("docs").resolve()
        self._load_registry()
        
    def _get_relative_path(self, file_path: str) -> str:
        """Convert absolute path to relative path for consistent registry keys"""
        # Normalize path separators to match registry format
        relative_path = str(Path(file_path).resolve().relative_to(self.base_path))
        return relative_path.replace('/', '\\')  # Convert to Windows-style paths to match registry
    
    def _get_full_path(self, relative_path: str) -> str:
        """Convert relative path to full path"""
        return str(self.base_path / relative_path)
    
    def _load_registry(self):
        """Load document processing registry"""
        try:
            with open(self.registry_path, 'r') as f:
                self.document_registry = json.load(f)
                print(f"Loaded existing registry with {len(self.document_registry)} entries")
        except FileNotFoundError:
            print("No existing registry found, creating new one")
            self.document_registry = {}
            
    def _check_document_processed(self, file_path: str) -> bool:
        """Check if document needs processing"""
        relative_path = self._get_relative_path(file_path)
        print(f"Checking path: {relative_path}")  # Debug print
        print(f"In registry: {relative_path in self.document_registry}")  # Debug print
        
        if relative_path not in self.document_registry:
            return False
            
        try:
            current_hash = self._get_document_hash(file_path)
            matches = self.document_registry[relative_path]['hash'] == current_hash
            print(f"Hash comparison for {relative_path}:")
            print(f"Current: {current_hash}")
            print(f"Registry: {self.document_registry[relative_path]['hash']}")
            print(f"Matches: {matches}")
            return matches
        except (FileNotFoundError, KeyError) as e:
            print(f"Error checking {relative_path}: {str(e)}")
            return False
    
    def _save_registry(self):
        """Save document processing registry"""
        print(f"Saving registry to {self.registry_path}")
        print(f"Registry contains {len(self.document_registry)} entries")
        with open(self.registry_path, 'w') as f:
            json.dump(self.document_registry, f, indent=2)
        print("Registry saved successfully")
    
    def _get_document_hash(self, file_path: str) -> str:
        """Get document hash based on content and modification time"""
        mtime = os.path.getmtime(file_path)
        with open(file_path, 'rb') as f:
            content = f.read()
        # Use SHA-256 instead of Python's hash()
        content_hash = hashlib.sha256(content).hexdigest()
        return f"{content_hash}_{mtime}"
    
    def process_documents(self, documents: Union[str, List[str], List['Document'], 'Generator'], 
                         force_reload: bool = False,
                         batch_size: int = 1,
                         rate_limit_delay: float = 2.0):
        """Process documents with rate limiting"""
        if isinstance(documents, (str, Path)):
            documents = [documents]
        
        # Convert generator to list if needed
        if hasattr(documents, '__iter__') and not isinstance(documents, (list, str)):
            documents = list(documents)
        
        # If documents are already Document objects
        if documents and hasattr(documents[0], 'page_content'):
            print(f"Total documents to check: {len(documents)}")
            docs_to_process = []
            
            # Check which documents need processing before splitting
            for doc in documents:
                if 'source' in doc.metadata:
                    relative_path = self._get_relative_path(doc.metadata['source'])
                    if force_reload or not self._check_document_processed(doc.metadata['source']):
                        docs_to_process.append(doc)
                    else:
                        print(f"Skipping {relative_path} - already processed")
            
            if not docs_to_process:
                print("All documents are up to date in the vector store.")
                return self._load_vector_store()
            
            print(f"Documents requiring processing: {len(docs_to_process)}")
            
            # Only split documents that need processing
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            split_documents = text_splitter.split_documents(docs_to_process)
            print(f"Total chunks to process: {len(split_documents)}")
            
            # Process in batches with rate limiting
            vector_store = self._load_vector_store()
            if vector_store is None:
                vector_store = FAISS.from_documents(split_documents[:batch_size], self.embeddings)
                # Process remaining documents with delay
                for i in range(batch_size, len(split_documents), batch_size):
                    time.sleep(rate_limit_delay)
                    batch = split_documents[i:i + batch_size]
                    print(f"Processing batch {i//batch_size + 1}/{(len(split_documents)-1)//batch_size + 1}")
                    vector_store.add_documents(batch)
            else:
                for i in range(0, len(split_documents), batch_size):
                    if i > 0:
                        time.sleep(rate_limit_delay)
                    batch = split_documents[i:i + batch_size]
                    print(f"Processing batch {i//batch_size + 1}/{(len(split_documents)-1)//batch_size + 1}")
                    vector_store.add_documents(batch)
            
            # Update registry and save
            for doc in docs_to_process:
                if 'source' in doc.metadata:
                    relative_path = self._get_relative_path(doc.metadata['source'])
                    self.document_registry[relative_path] = {
                        'hash': self._get_document_hash(doc.metadata['source']),
                        'last_processed': datetime.now().isoformat()
                    }
            
            vector_store.save_local(self.index_name)
            self._save_registry()
            return vector_store
        
        # Original file path processing logic
        docs_to_process = []
        for file_path in documents:
            full_path = self._get_full_path(file_path)
            if force_reload or not self._check_document_processed(full_path):
                docs_to_process.append(full_path)
            else:
                print(f"Skipping {file_path} - already processed")
        
        if not docs_to_process:
            print("All documents are up to date in the vector store.")
            return self._load_vector_store()
            
        # Process only new or modified documents
        documents = []
        for file_path in docs_to_process:
            print(f"Processing {file_path}...")
            loader = TextLoader(file_path, encoding='utf-8')
            doc = loader.load()
            
            doc_hash = self._get_document_hash(file_path)
            for d in doc:
                d.metadata.update({
                    'source': file_path,
                    'filename': Path(file_path).name,
                    'directory': str(Path(file_path).parent),
                    'processed_at': datetime.now().isoformat(),
                    'doc_hash': doc_hash
                })
            documents.extend(doc)
            
            # Update registry with relative path
            relative_path = self._get_relative_path(file_path)
            self.document_registry[relative_path] = {
                'hash': doc_hash,
                'last_processed': datetime.now().isoformat()
            }
        
        # Split and embed new documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        split_documents = text_splitter.split_documents(documents)
        
        # Process in batches with rate limiting
        vector_store = self._load_vector_store()
        if vector_store is None:
            vector_store = FAISS.from_documents(split_documents[:batch_size], self.embeddings)
            # Process remaining documents with delay
            for i in range(batch_size, len(split_documents), batch_size):
                time.sleep(rate_limit_delay)  # Add delay between batches
                batch = split_documents[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(split_documents)-1)//batch_size + 1}")
                vector_store.add_documents(batch)
        else:
            for i in range(0, len(split_documents), batch_size):
                if i > 0:  # Don't delay first batch
                    time.sleep(rate_limit_delay)
                batch = split_documents[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(split_documents)-1)//batch_size + 1}")
                vector_store.add_documents(batch)
        
        # Save updates
        vector_store.save_local(self.index_name)
        self._save_registry()
        
        return vector_store
    
    def _load_vector_store(self):
        """Load existing vector store if available"""
        if os.path.exists(self.index_name):
            return FAISS.load_local(
                self.index_name,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        return None

def create_rag_system(file_path: str):
    # Load single markdown file
    loader = TextLoader(file_path)
    doc = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    documents = text_splitter.split_documents(doc)
    
    # Define the embedding model - Note: Still using Mistral API for embeddings
    # as local models don't support embeddings yet
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=os.getenv("MISTRAL_API_KEY")
    )
    
    # Create the vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Save the vector store with a name based on the input file
    index_name = f"faiss_index_{os.path.basename(file_path)}"
    vector_store.save_local(index_name)
    
    return vector_store, index_name

def setup_qa_chain(vector_store, local_model: LocalMistralChat):
    # Define retriever
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}  # Return top 3 most relevant chunks
    )
    
    # Define prompt template
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}

Please provide a clear and concise answer based solely on the information found in the context above.
If the answer cannot be found in the context, please respond with "I cannot answer this question based on the provided context."
""")
    
    def generate_response(prompt_value):
        return local_model.generate(prompt_value)
    
    # Create custom chain that uses local model
    def process_query(input_dict):
        context = input_dict["context"]
        question = input_dict["input"]
        prompt_value = prompt.format(context=context, input=question)
        return {"answer": generate_response(prompt_value)}
    
    def run_chain(input_dict):
        # Get relevant documents using invoke instead of get_relevant_documents
        docs = retriever.invoke(input_dict["input"])
        # Format documents
        context = "\n\n".join(doc.page_content for doc in docs)
        # Generate response
        return process_query({"context": context, "input": input_dict["input"]})
    
    return run_chain

def query_docs(chain, query: str):
    response = chain({"input": query})
    return response["answer"]

if __name__ == "__main__":
    # Initialize embeddings and local model
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=os.getenv("MISTRAL_API_KEY")
    )
    local_model = LocalMistralChat(model, tokenizer)
    
    # Initialize lazy RAG system
    rag_system = LazyRAGSystem(embeddings)
    
    # Use DirectoryLoader with UTF-8 encoding
    loader = DirectoryLoader(
        "docs",
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},  # Add UTF-8 encoding
        show_progress=True,
        #use_multithreading=True
    )
    
    # Process documents with rate limiting
    vector_store = rag_system.process_documents(
        list(loader.lazy_load()),
        batch_size=1,  # Smaller batches
        rate_limit_delay=2.0  # 1 second delay between batches
    )
    
    # Setup QA chain
    qa_chain = setup_qa_chain(vector_store, local_model)
    
    # Interactive query loop
    print("Enter your questions (type 'exit' to quit):")
    while True:
        query = input("\nQ: ")
        if query.lower() == 'exit':
            break
        answer = query_docs(qa_chain, query)
        print(f"A: {answer}") 