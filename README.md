# PDF Insights AI

**Unlock knowledge from documents with intelligent search and insights**

**PDF Insights AI** is an intelligent chatbot that allows users to interact with PDF documents in a conversational manner. Instead of manually searching through lengthy files, users can simply ask questions and instantly receive precise, context-aware answers.

## How It Works
1. **Upload PDF** – Extract text from PDF using **PyMuPDF**  
2. **Chunking** – Split text into small, manageable pieces using **Regex**  
3. **Embedding** – Convert chunks into vector embeddings using **SentenceTransformers**  
4. **Indexing** – Store embeddings in **FAISS** for efficient similarity search  
5. **Retrieval** – Retrieve the most relevant chunks when a user asks a question  
6. **Response Generation** – Pass query + retrieved chunks to **LLM** for natural responses  
7. **UI & Chat History** – Displays conversational responses in **Streamlit UI**

## Tech Stack
- **Frontend/UI:** Streamlit (session-based chat interface)  
- **Document Processing:** PyMuPDF, Regex  
- **Vector Search:** FAISS, SentenceTransformers  
- **LLM:** Google Gemini

## Access & Links
- **Demo Video:** [Click Here](https://drive.google.com/file/d/1a_zZd1rwCjoju7NFPxv0PHR5DMA9Tp1L/view?usp=drivesdk)  
- **GitHub Repository:** [PDF Insights AI](https://github.com/RenukaBondapalli/PDF-Insights-AI.git)  
- **Try It Live:** [PDF Insights AI App](https://pdfinsights-ai.streamlit.app/)
