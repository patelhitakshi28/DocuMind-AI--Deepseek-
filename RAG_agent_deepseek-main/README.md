# 📘 IntelliDoc AI

Your Intelligent Document Assistant

## 🚀 Overview

IntelliDoc AI is an AI-powered document assistant that allows users to upload PDF and ask questions about their content. It utilizes LangChain, DeepSeek's LLM, and Streamlit to provide a smooth and interactive experience for document-based Q&A.

## 🔧 Features

📂 Upload & Process PDFs

🔍 Extract and Index Text

🤖 Chat with Documents using AI

📝 Chat History to keep track of past questions

⏳ Real-time AI Responses

🏗 Uses DeepSeek RAG Model for Answers

🎨 Dark-themed UI for better readability

## 📦 Installation

1️⃣ Clone the Repository

git clone https://github.com/YOUR_USERNAME/IntelliDoc-AI.git
cd IntelliDoc-AI

2️⃣ Install Dependencies

Make sure you have Python 3.8+ installed.

pip install -r requirements.txt

3️⃣ Run the Application

streamlit run app.py

## 🛠 Technologies Used

Python 🐍

Streamlit 🎨 - UI Framework

LangChain 🔗 - Document Processing & LLM Integration

DeepSeek AI 🧠 - AI Language Model

PDFPlumber 📄 - PDF Parsing

## 📖 Usage

Run the App and open it in your browser.

Upload a PDF using the file uploader.

Ask questions about the document in the chat input.

View answers and chat history dynamically.



## 🛑 Troubleshooting

Q: My PDF is not being processed!✅ Ensure the file path is correct in the PDF_STORAGE_PATH variable.

Q: Not able to Upload Pdf (AxiosError: Request failed with status code 403)✅ Then run the main file using this command: streamlit run rag_deep_main.py --server.enableXsrfProtection=false --server.enableCORS=false

Q: The AI is not responding!✅ Check if Ollama and DeepSeek models are installed and running:

ollama list
ollama serve
ollama pull deepseek-r1:1.5b

Q: How do I clear chat history?✅ Restart the Streamlit app or manually reset st.session_state.chat_history.


