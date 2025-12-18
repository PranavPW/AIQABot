# AIQAbot (RAG Application)

A Retrieval-Augmented Generation (RAG) chatbot application using LangChain, OpenAI, ChromaDB, and Gradio. This application allows users to update the knowledge base with PDF documents and query it using a conversational interface.

## Prerequisites

- **Python 3.12** is required.
- An OpenAI API Key.

## Installation

1. **Clone or Download** the repository.
2. **Set up a Virtual Environment**:
   It is recommended to use a virtual environment to manage dependencies.
   ```bash
   py -3.12 -m venv venv_3.12
   ```
3. **Activate the Environment**:
   - Windows:
     ```bash
     venv_3.12\Scripts\activate
     ```
4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

Double-click the `run_app.bat` file, or run the following command in your terminal (ensure your virtual environment is active):

```bash
python app.py
```

The application will launch in your default web browser (usually at `http://127.0.0.1:7860`).

### Features

- **Chat Interface**: Ask questions to the AI about the uploaded documents.
- **Dynamic Configuration**: Change the LLM Model (e.g., gpt-3.5-turbo, gpt-4) and Embedding Model via the implementation settings in the code or UI (if configured).
- **Document Upload**: Add new PDFs to the knowledge base directly from the UI.

## Notes

- The project ignores various debug and verification scripts (`debug_*.py`, `verify_*.py`) to keep the repository clean.
