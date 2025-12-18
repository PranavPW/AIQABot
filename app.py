import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Global cache for vector stores
# Key: file_path, Value: items associated with that file (e.g., vectordb or retriever)
vector_store_cache = {}

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
def get_embedding_model(base_url, model_name):
    """
    Initializes and returns an OllamaEmbeddings instance.
    Notes: Ollama base URL usually doesn't need /v1 for native embedding endpoints, 
    but LangChain's OllamaEmbeddings class handles base_url standardly (often e.g. http://localhost:11434).
    The user provided input typically has /v1. We might need to strip or adjust if issues arise.
    Standard Ollama URL: http://localhost:11434
    """
    # Simple check: if the user provides a /v1 url for OpenAI compatibility, 
    # OllamaEmbeddings often defaults to native API. 
    # We will strip '/v1' just in case if the user provides it, to be safe for the native wrapper.
    clean_base_url = base_url.replace("/v1", "") if base_url.endswith("/v1") else base_url
    
    embeddings = OllamaEmbeddings(
        base_url=clean_base_url,
        model=model_name
    )
    return embeddings

## Vector db with Caching
def get_vector_store(file_path, embedding_model):
    """
    Creates or retrieves a cached Chroma vector database from text chunks.
    Args:
        file_path: Path to the pdf file.
        embedding_model: The embedding model instance.
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
def retriever_qa(file, query, base_url, api_key, llm_model_name, embedding_model_name):
    """
    Performs a question-answering task using the RAG chain.
    """
    try:
        if file is None:
            return "Please upload a PDF file first."

        # Initialize models
        llm = get_llm(base_url, api_key, llm_model_name)
        embedding_model = get_embedding_model(base_url, embedding_model_name)
        
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
        return f"Error occurred: {str(e)}\n\nPlease ensure your Ollama server is running and the models ({llm_model_name}, {embedding_model_name}) are installed."

# Create Gradio interface
with gr.Blocks() as rag_application:
    gr.Markdown("# Universal RAG Chatbot")
    gr.Markdown("Upload a PDF document and ask any question. Configure your local LLM settings below.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath")
            
            with gr.Accordion("Advanced Settings", open=False):
                base_url_input = gr.Textbox(label="Base URL", value="http://localhost:11434/v1")
                api_key_input = gr.Textbox(label="API Key", value="ollama")
                llm_model_input = gr.Textbox(label="LLM Model", value="qwen2.5:latest")
                embedding_model_input = gr.Textbox(label="Embedding Model", value="nomic-embed-text")
        
        with gr.Column(scale=2):
            query_input = gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
            output_text = gr.Textbox(label="Output", lines=10)
            submit_btn = gr.Button("Submit")
            clear_btn = gr.Button("Clear Output")

    submit_btn.click(
        fn=retriever_qa,
        inputs=[
            file_input, 
            query_input, 
            base_url_input, 
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
