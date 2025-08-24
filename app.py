import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import re

# Load environment variables
load_dotenv()
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

# Extract Text
def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page_num, page in enumerate(doc):
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.get_text("text")
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# Chunking
def chunk_text(text, chunk_size=500):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

# Embeddings
def create_faiss_index(chunks, embed_model):
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# Retrieve 
def retrieve_context(query, chunks, index, embed_model, top_k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    _, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0]], indices[0]

#LLM
def ask_gemini(question, context, chat_history=None):
    # Handle greetings & casual chat
    greetings = ["hi", "hello", "hey", "hii", "good morning", "good evening", "what's up", "what are you doing"]
    if any(g in question.lower() for g in greetings):
        return "Hii 👋 I’m ready! Upload a document and let’s chat about it 📄"

    history_context = ""
    if chat_history:
        history_context = "\nPrevious conversation:\n"
        for msg in chat_history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_context += f"{role}: {msg['content']}\n"
    
    prompt = f"""
You are an expert document analysis assistant. Use the provided context from the user's documents to answer their query accurately and helpfully.

{history_context}

Document Context:
{context}

User Question:
{question}

Instructions:
1. Answer based ONLY on the provided document context
2. If the context doesn't contain the answer, say "The documents don't contain information to answer this question"
3. Be precise and cite information directly from the context when possible
4. For multi-part questions, answer each part clearly
5. Format your response in a clear, readable way

Answer:
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"
    

# Custom CSS 
def apply_custom_css():
    st.markdown("""
<style>
}

/* ----------------- Chat input ----------------- */
/* Container */
.stChatInput {
    position: fixed;
    bottom: 20px;
    width: 70%;
    background: rgba(30, 64, 175, 0.2);
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    padding: 0.5rem;
}

/* ----------------- Buttons ----------------- */
.stButton button {
    background-color: #1e40af;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: 500;
}

.stButton button:hover {
    background-color: #2563eb;
}

/* ----------------- File uploader ----------------- */
.stFileUploader {
    border: 2px dashed #d1d5db;
    border-radius: 8px;
    padding: 1rem;
    background-color: rgba(30, 64, 175, 0.2); /* faint transparent blue */
    color: white;
}

</style>

    """, unsafe_allow_html=True)

apply_custom_css()


# Streamlit UI
st.set_page_config(page_title="PDF-Insights-AI", layout="wide")

st.markdown(
    """
    <div style='
        background: linear-gradient(135deg, #4A90E2, #2013FE);
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0px 6px 15px rgba(0,0,0,0.2);
        text-align: center;
    '>
        <h1 style='color: #ffffff; margin-bottom: 12px; font-size: 42px;'>DocuQuery AI</h1>
        <p style='color: #f0f0f0; font-size:20px; margin: 0;'>
            Unlock knowledge from documents with intelligent search and insights.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "embed_model" not in st.session_state:
    st.session_state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "last_response_context" not in st.session_state:
    st.session_state.last_response_context = None

# Sidebar: PDF upload & settings
with st.sidebar:
    st.header("📂 Upload PDFs")
    uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    
    # Display currently processed files
    if st.session_state.processed_files:
        st.subheader("📋 Currently Processed Documents")
        for i, file_name in enumerate(st.session_state.processed_files, 1):
            st.write(f"{i}. {file_name}")

    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk Size", min_value=200, max_value=1000, value=500, step=50)
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3, step=1)
    
    clear_btn = st.button("🗑️ Clear Chat")

    if clear_btn:
        st.session_state.chunks = []
        st.session_state.index = None
        st.session_state.processed_files = []
        st.session_state.chat_history = []
        st.session_state.last_response_context = None
        st.rerun()

# Automatic document processing 
if uploaded_files and uploaded_files != st.session_state.get('last_uploaded_files', []):
    # Store the current uploaded files to avoid reprocessing
    st.session_state.last_uploaded_files = uploaded_files
    
    with st.sidebar:
        with st.spinner("Processing documents..."):
            all_text = ""
            file_names = []
            for uploaded_file in uploaded_files:
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    all_text += text + "\n"
                    file_names.append(uploaded_file.name)
            
            if all_text.strip():
                st.session_state.chunks = chunk_text(all_text, chunk_size=chunk_size)
                st.session_state.index, _ = create_faiss_index(st.session_state.chunks, st.session_state.embed_model)
                st.session_state.processed_files = file_names
                st.success(f"✅ {len(uploaded_files)} file(s) processed. {len(st.session_state.chunks)} chunks created. Ready to chat!")
            else:
                st.error("No text could be extracted from the uploaded documents.")


# Main chat area

st.subheader("💬 Chat with your documents")

# Display document status
if st.session_state.processed_files:
    st.info(f"📄 Analyzing {len(st.session_state.processed_files)} document(s)")
else:
    st.info("👈 Upload PDFs in the sidebar to begin")

# Container for chat messages
chat_container = st.container()

# Display chat history
with chat_container:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            # User message
            col1, col2 = st.columns([0.7, 0.3])
            with col2:
                with st.chat_message("user"):
                    st.markdown(msg["content"])
        else:
            # Assistant message 
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

                # Show source chunks if available
                if "source_chunks" in msg:
                    with st.expander("🔍 View Source Chunks", expanded=False):
                        st.caption("Document chunks used for this response:")
                        for i, (chunk, idx) in enumerate(zip(msg["source_chunks"], msg["chunk_indices"])):
                            # Extract page number if available
                            page_match = re.search(r'--- Page (\d+) ---', chunk)
                            page_info = f" (Page {page_match.group(1)})" if page_match else ""

                            with st.expander(f"Source {i+1}{page_info}", expanded=False):
                                st.info(chunk[:500] + "..." if len(chunk) > 500 else chunk)

# Chat input
user_input = st.chat_input("Ask a question about your PDFs...", key="chat_input_bottom")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # render the user message 
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)

    if st.session_state.index is not None:
        with st.spinner("🔍 Searching documents..."):
            context_chunks, chunk_indices = retrieve_context(
                user_input,
                st.session_state.chunks,
                st.session_state.index,
                st.session_state.embed_model,
                top_k=top_k
            )
        context_text = "\n".join(context_chunks)
    else:
        context_text = ""
        context_chunks = []
        chunk_indices = []

    with st.spinner("💭 Generating response..."):
        response = ask_gemini(user_input, context_text, st.session_state.chat_history)

    # Add assistant response to chat history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "source_chunks": context_chunks,
        "chunk_indices": chunk_indices
    })

    # Force rerun
    st.rerun()

# some helpful tips
with st.expander("💡 Tips for better results"):
    st.markdown("""
    - Ask specific questions rather than general ones
    - Reference page numbers if you know where information is located
    - For complex questions, break them down into multiple simpler questions
    - The chatbot can only answer based on the documents you've uploaded
    - Larger chunk sizes may work better for technical documents

    """)

