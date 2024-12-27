import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
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

# Load environment variables, prioritizing .env.local
load_dotenv(dotenv_path='.env.local')  # Try .env.local first
if not os.getenv("MISTRAL_API_KEY"):  # If API key not found, try .env as fallback
    load_dotenv(dotenv_path='.env')

# Path to your local Mistral model
MISTRAL_MODEL_PATH = r"E:\mistral_models\mistral-7B-Instruct-v0.3"

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
    # Example usage - specify your markdown file path
    file_path = "contributing.md"  # Update this to your markdown file path
    
    # Initialize local model
    local_model = LocalMistralChat(model, tokenizer)
    
    # Generate index name for checking existence
    index_name = f"faiss_index_{os.path.basename(file_path)}"
    
    # Create or load vector store
    if os.path.exists(index_name):
        vector_store = FAISS.load_local(
            index_name, 
            MistralAIEmbeddings(mistral_api_key=os.getenv("MISTRAL_API_KEY")),
            allow_dangerous_deserialization=True  # Only use this if you trust the source of the index
        )
    else:
        vector_store, _ = create_rag_system(file_path)
    
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