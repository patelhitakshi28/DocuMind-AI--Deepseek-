import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Streamlit UI Styling
st.markdown("""
    <style>
    /* Main App Background */
    .stApp {
        background-color: #1A1A2E;  /* Deep dark violet */
        color: #E0E0E0;  /* Light text for readability */
    }

    /* Header Styling */
    h1, h2, h3 {
        color: #BB86FC !important;  /* Light violet for headers */
    }

    /* Chat Input Styling */
    .stChatInput input {
        background-color: #2A2A3D !important;  /* Dark violet input background */
        color: #E0E0E0 !important;  /* Light text */
        border: 1px solid #BB86FC !important;  /* Violet border */
        border-radius: 10px !important;
        padding: 10px !important;
    }

    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #2A2A3D !important;  /* Dark violet for user messages */
        border: 1px solid #BB86FC !important;  /* Violet border */
        color: #E0E0E0 !important;  /* Light text */
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }

    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #3A3A4D !important;  /* Slightly lighter violet for assistant messages */
        border: 1px solid #BB86FC !important;  /* Violet border */
        color: #E0E0E0 !important;  /* Light text */
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }

    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #BB86FC !important;  /* Violet avatar */
        color: #1A1A2E !important;  /* Dark text for contrast */
    }

    /* File Uploader Styling */
    .stFileUploader {
        background-color: #2A2A3D !important;  /* Dark violet background */
        border: 1px solid #BB86FC !important;  /* Violet border */
        border-radius: 10px;
        padding: 15px;
    }

    /* Button Styling */
    .stButton button {
        background-color: #BB86FC !important;  /* Violet button */
        color: #1A1A2E !important;  /* Dark text */
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
    }

    /* Spinner Styling */
    .stSpinner div {
        color: #BB86FC !important;  /* Violet spinner */
    }

    /* Success Message Styling */
    .stSuccess {
        background-color: #2A2A3D !important;  /* Dark violet background */
        border: 1px solid #BB86FC !important;  /* Violet border */
        color: #E0E0E0 !important;  /* Light text */
        border-radius: 10px;
        padding: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Constants
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If the answer can be provided in one word or a short phrase, do so. Only provide a detailed explanation if absolutely necessary.
Do not show your reasoning steps or thinking process. Provide only the final answer.

Query: {user_query} 
Context: {document_context} 
Answer:
"""

PDF_STORAGE_PATH = r"C:\Users\admin\OneDrive\Desktop\Projects\RAG_agent_deepseek-main\RAG_agent_deepseek-main\Documents"
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b", max_tokens=50, temperature=0.3)  # Lower temperature for precision

# Functions
def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def clean_response(response):
    """
    Removes unwanted tags or reasoning steps from the model's response.
    """
    if "<think>" in response:
        # Extract only the final answer after the last </think> tag
        response = response.split("</think>")[-1].strip()
    return response

def simplify_response(response):
    """
    Simplifies the response to a minimal form if possible.
    """
    sentences = response.split(". ")
    if len(sentences) > 0:
        first_sentence = sentences[0]
        if len(first_sentence.split()) <= 5:  # If the first sentence is short, return it
            return first_sentence
    return response  # Fallback to the original response

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    response = response_chain.invoke({"user_query": user_query, "document_context": context_text})
    
    # Clean the response to remove unwanted tags or reasoning steps
    cleaned_response = clean_response(response)
    
    # Simplify the response if possible
    simplified_response = simplify_response(cleaned_response)
    return simplified_response

# UI Configuration
st.title("📘 DocuMind AI")
st.markdown("### Exploring Text, Discovering Answers")
st.markdown("---")

# Sidebar for Additional Options
with st.sidebar:
    st.markdown("### 🛠️ Settings")
    response_length = st.slider("Response Length", 1, 100, 50, help="Adjust the length of the model's responses.")
    st.markdown("---")
    st.markdown("### 📂 Upload Document")
    uploaded_pdf = st.file_uploader(
        "Upload a PDF",
        type="pdf",
        help="Select a PDF document for analysis",
        accept_multiple_files=False
    )
    st.markdown("---")
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

# Welcome Message
if "chat_history" not in st.session_state or len(st.session_state.chat_history) == 0:
    st.markdown("""
    ### 👋 Welcome to IntelliDoc AI!
    **How to use:**
    1. Upload a PDF document using the sidebar.
    2. Ask questions about the document in the chat below.
    3. Enjoy concise and accurate answers!
    """)

# File Processing
if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    st.success("✅ Document processed successfully! Ask your questions below.")

# Chat History Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat Input
user_input = st.chat_input("Enter your question about the document...")

if user_input:
    st.session_state.chat_history.append(("User", user_input))
    
    with st.spinner("Analyzing document..."):
        relevant_docs = find_related_documents(user_input)
        ai_response = generate_answer(user_input, relevant_docs)

    st.session_state.chat_history.append(("AI", ai_response))

# Display Chat History
st.markdown("## 📝 Chat History")
for role, text in st.session_state.chat_history:
    with st.chat_message(role.lower()):
        st.write(text)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #BB86FC;">
        <p>Powered by Deepseek | © 2023 IntelliDoc AI</p>
    </div>
    """, unsafe_allow_html=True)