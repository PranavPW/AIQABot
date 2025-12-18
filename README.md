AIQAbot (RAG Application)

AIQAbot is a Retrieval-Augmented Generation (RAG) chatbot built with LangChain, ChromaDB, and Gradio. It allows you to transform your PDF documents into an interactive knowledge base powered by local LLMs.
🚀 Key Features

    Local LLM Support: Powered by Ollama for maximum data privacy.

    Optimized Model: Pre-configured for qwen3:4b for high-quality, refined results.

    Dynamic Updates: Upload PDFs via the UI to update the knowledge base instantly.

    Network Access: Accessible by other devices on your local network.

🛡️ Privacy & Security (The Self-Hosting Advantage)

This project is designed for users who handle sensitive information (legal docs, medical records, or proprietary research) and cannot risk uploading data to cloud providers like OpenAI or Anthropic.
Why Self-Host AIQAbot?

    Zero Data Leakage: Your documents never leave your physical machine. Processing happens entirely on your local hardware.

    No Training Usage: Unlike many cloud APIs, your data is never used to "improve" or train future versions of the AI model.

    Air-Gap Compatibility: Because it uses Ollama and ChromaDB locally, this app can be configured to run in environments without an internet connection.

    Data Sovereignty: You own your database (ChromaDB) and your model weights. You aren't at the mercy of a third-party's privacy policy changes.

🛠️ Prerequisites

    Python: 3.12

    Ollama: The latest desktop application for Windows or macOS.

📦 Installation & Setup
1. Clone & Environment
Bash

git clone https://github.com/PranavPW/AIQAbot.git
cd AIQAbot
py -3.12 -m venv venv_3.12

2. Activate & Install

    Windows: venv_3.12\Scripts\activate

    Linux/Mac: source venv_3.12/bin/activate

    Install Dependencies: pip install -r requirements.txt

🧠 Ollama Configuration (GUI Method)
1. Download the Model

Pull the recommended default model via your terminal:
Bash

ollama pull qwen3:4b

2. Enable Network Access (GUI)

    Open the Ollama App from your system tray.

    Go to Settings (Gear icon).

    Locate the toggle "Expose Ollama to the network".

    Switch it to ON. This allows AIQAbot to communicate with Ollama on port 11434.

🏃 Usage
Running the Application
Bash

python app.py

Accessing the UI

    On this PC: http://127.0.0.1:7860

    From another device: http://<your-local-ip-address>:7860

🤝 Support & Contribution

If you find this project helpful, any form of support is greatly appreciated!

    Star the Repo: Give it a ⭐ on GitHub!

    Feedback: Open an issue for bugs or suggestions.

    Contribute: Pull requests are always welcome!

📄 License

This project is licensed under the MIT License.