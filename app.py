import gradio as gr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import warnings
import os
import requests

# Suppress warnings
warnings.filterwarnings('ignore')

# Global cache for vector stores
# Key: file_path, Value: items associated with that file (e.g., vectordb or retriever)
vector_store_cache = {}

## Model Fetching Helpers
def get_ollama_models(base_url):
    """Fetch models from Ollama /api/tags"""
    try:
        if not base_url.startswith("http"):
            base_url = f"http://{base_url}"
        
        # Strip /v1 if present for native tagging
        tag_url = base_url.replace("/v1", "") if base_url.endswith("/v1") else base_url
        response = requests.get(f"{tag_url}/api/tags")
        if response.status_code == 200:
            data = response.json()
            # Ollama models are usually in 'models' key with 'name' field
            return [model['name'] for model in data.get('models', [])]
        return []
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return []

def get_openai_models(base_url, api_key):
    """Fetch models from OpenAI-compatible /v1/models"""
    try:
        if not base_url.startswith("http"):
            base_url = f"http://{base_url}"
            
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(f"{base_url}/models", headers=headers)
        if response.status_code == 200:
            data = response.json()
            return [model['id'] for model in data.get('data', [])]
        return []
    except Exception as e:
        print(f"Error fetching OpenAI/LM Studio models: {e}")
        return []

def refresh_models(provider, host, port, api_key):
    """
    Constructs URL and fetches models based on provider.
    Returns updates for both LLM and Embedding dropdowns.
    """
    base_url = f"http://{host}:{port}"
    models = []
    
    if provider == "Ollama":
        models = get_ollama_models(base_url)
    elif provider in ["LM Studio", "OpenAI"]:
        # For LM Studio, usually http://localhost:1234/v1
        full_url = f"{base_url}/v1"
        models = get_openai_models(full_url, api_key)
    
    # If no models found, return empty list or keep current if possible, 
    # but here we just return what we found.
    # We return the same list for both LLM and Embedding for simplicity, 
    # though usually embedding models are distinct. User selects from the same list.
    
    if not models:
        return gr.update(choices=[]), gr.update(choices=[])
        
    return gr.update(choices=models, value=models[0] if models else None), \
           gr.update(choices=models, value=models[0] if models else None)


## LLM
def get_llm(base_url, api_key, model_name):
    """
    Initializes and returns a ChatOpenAI instance for text generation.
    """
    llm = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        temperature=0.5
    )
    return llm

## Document loader
def document_loader(file_path):
    """
    Loads a PDF document from the given file path.
    Args:
        file_path: The absolute path to the PDF file (string).
    Returns:
        A list of loaded documents.
    """
    loader = PyPDFLoader(file_path)
    loaded_document = loader.load()
    return loaded_document

## Text splitter
def text_splitter(data):
    """
    Splits the loaded documents into smaller chunks for processing.
    Args:
        data: A list of documents to be split.
    Returns:
        A list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = splitter.split_documents(data)
    return chunks

## Embedding model
def get_embedding_model(base_url, model_name, provider, api_key):
    """
    Initializes and returns an embedding model instance.
    """
    if provider == "Ollama":
        # Simple check: if the user provides a /v1 url for OpenAI compatibility, 
        # OllamaEmbeddings often defaults to native API. 
        # We will strip '/v1' just in case if the user provides it, to be safe for the native wrapper.
        clean_base_url = base_url.replace("/v1", "") if base_url.endswith("/v1") else base_url
        
        embeddings = OllamaEmbeddings(
            base_url=clean_base_url,
            model=model_name
        )
    else:
        # For LM Studio (OpenAI compatible)
        embeddings = OpenAIEmbeddings(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            check_embedding_ctx_length=False
        )
    return embeddings

## Vector db with Caching
def get_vector_store(file_path, embedding_model):
    """
    Creates or retrieves a cached Chroma vector database from text chunks.
    Args:
        file_path: Path to the pdf file.
    Returns:
        A Chroma vector database instance.
    """
    if file_path in vector_store_cache:
        print(f"Using cached vector store for: {file_path}")
        return vector_store_cache[file_path]
    
    print(f"Generating new vector store for: {file_path}")
    docs = document_loader(file_path)
    chunks = text_splitter(docs)
    vectordb = Chroma.from_documents(chunks, embedding_model)
    
    vector_store_cache[file_path] = vectordb
    return vectordb

## QA Chain
def retriever_qa(file, query, provider, host, port, api_key, llm_model_name, embedding_model_name):
    """
    Performs a question-answering task using the RAG chain.
    """
    try:
        if file is None:
            return "Please upload a PDF file first."

        # Construct Base URL
        base_url = f"http://{host}:{port}"
        if provider != "Ollama": # appends /v1 for OpenAI compatible
             base_url = f"{base_url}/v1"
        
        # Initialize models
        llm = get_llm(base_url, api_key, llm_model_name)
        embedding_model = get_embedding_model(base_url, embedding_model_name, provider, api_key)
        
        # Get Vector Store (Cached)
        vectordb = get_vector_store(file, embedding_model)
        retriever_obj = vectordb.as_retriever()
        
        # Create a prompt template for the final answer
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # Create the stuff documents chain (LLM + Prompt)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create the retrieval chain (Retriever + Question Answer Chain)
        rag_chain = create_retrieval_chain(retriever_obj, question_answer_chain)
        
        # Invoke the chain
        response = rag_chain.invoke({"input": query})
        
        return response['answer']
        
    except Exception as e:
        return f"Error occurred: {str(e)}\n\nPlease ensure your Server is running at http://{host}:{port} and models are available."

def update_port(provider):
    if provider == "Ollama":
        return "11434"
    elif provider == "LM Studio":
        return "1234"
    return "8000"

# Create Gradio interface
with gr.Blocks() as rag_application:
    gr.Markdown("# Universal RAG Chatbot")
    gr.Markdown("Upload a PDF document and ask any question. Configure your local LLM settings below.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath")
            
            with gr.Accordion("Model Settings", open=True):
                with gr.Row():
                    provider_input = gr.Dropdown(choices=["Ollama", "LM Studio", "OpenAI"], value="Ollama", label="Provider")
                    host_input = gr.Textbox(label="Host", value="localhost")
                    port_input = gr.Textbox(label="Port", value="11434")
                
                api_key_input = gr.Textbox(label="API Key", value="ollama", type="password")
                refresh_btn = gr.Button("Refresh Models")
                
                llm_model_input = gr.Dropdown(label="LLM Model", choices=["qwen2.5:latest"], value="qwen2.5:latest", allow_custom_value=True)
                embedding_model_input = gr.Dropdown(label="Embedding Model", choices=["nomic-embed-text"], value="nomic-embed-text", allow_custom_value=True)

        with gr.Column(scale=2):
            query_input = gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
            output_text = gr.Textbox(label="Output", lines=10)
            submit_btn = gr.Button("Submit")
            clear_btn = gr.Button("Clear Output")

    # Event Listeners
    provider_input.change(fn=update_port, inputs=provider_input, outputs=port_input)
    
    refresh_btn.click(
        fn=refresh_models,
        inputs=[provider_input, host_input, port_input, api_key_input],
        outputs=[llm_model_input, embedding_model_input]
    )

    submit_btn.click(
        fn=retriever_qa,
        inputs=[
            file_input, 
            query_input, 
            provider_input,
            host_input,
            port_input,
            api_key_input, 
            llm_model_input, 
            embedding_model_input
        ],
        outputs=output_text
    )
    
    clear_btn.click(
        fn=lambda: ("", None),
        inputs=[],
        outputs=[output_text, file_input]
    )

# Launch the app
if __name__ == "__main__":
    rag_application.launch(server_name="0.0.0.0", server_port=7860)
