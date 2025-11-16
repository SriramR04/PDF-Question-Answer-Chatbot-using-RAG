import streamlit as st
import os
import shutil
from utils.pdf_processor import extract_text_from_pdf
from utils.embeddings import create_embeddings, query_embeddings
from utils.rag_chain import generate_answer

# Page configuration
st.set_page_config(
    page_title="PDF QA Chatbot",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None

def clear_session():
    """Clear all session data and ChromaDB"""
    st.session_state.chat_history = []
    st.session_state.pdf_processed = False
    st.session_state.current_pdf = None
    
    # Clear ChromaDB directory
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
    
    st.rerun()

def process_pdf(uploaded_file):
    """Process uploaded PDF and create embeddings"""
    # Check if a different PDF is being uploaded
    if st.session_state.current_pdf and st.session_state.current_pdf != uploaded_file.name:
        clear_session()
    
    # Save uploaded file temporarily
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # Extract text from PDF
        with st.spinner("Extracting text from PDF..."):
            text_content = extract_text_from_pdf(temp_path)
        
        if not text_content.strip():
            st.error("No text could be extracted from the PDF. Please ensure it contains readable text or scanned images.")
            return False
        
        # Create embeddings and store in ChromaDB
        with st.spinner("Creating embeddings and storing in vector database..."):
            create_embeddings(text_content)
        
        st.session_state.pdf_processed = True
        st.session_state.current_pdf = uploaded_file.name
        st.success(f"✅ PDF '{uploaded_file.name}' processed successfully! You can now ask questions.")
        return True
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Main UI
st.title("📄 PDF Question-Answer Chatbot")
st.markdown("Upload a PDF and ask questions about its content using RAG technology.")

# Sidebar
with st.sidebar:
    st.header("📤 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file (max 10 pages recommended)",
        type=['pdf'],
        help="Upload a PDF document to start asking questions"
    )
    
    if uploaded_file:
        if st.button("🔄 Process PDF", use_container_width=True):
            process_pdf(uploaded_file)
    
    st.divider()
    
    if st.session_state.pdf_processed:
        st.success(f"📄 Current PDF: {st.session_state.current_pdf}")
        if st.button("🗑️ Clear & Reset", use_container_width=True):
            clear_session()
    
    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown("""
    This chatbot uses:
    - **PyMuPDF** for text extraction
    - **Tesseract OCR** for scanned pages
    - **ChromaDB** for vector storage
    - **Llama-3.3-70B** via Groq for answers
    - **RAG** technology for accurate responses
    """)

# Main chat area
if st.session_state.pdf_processed:
    st.markdown("### 💬 Chat with your PDF")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if question := st.chat_input("Ask a question about the PDF..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve relevant chunks
                    relevant_chunks = query_embeddings(question, top_k=3)
                    
                    if not relevant_chunks:
                        answer = "I couldn't find relevant information in the PDF to answer your question."
                    else:
                        # Generate answer using RAG
                        answer = generate_answer(question, relevant_chunks)
                    
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"Error generating answer: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

else:
    st.info("👈 Please upload a PDF file from the sidebar and click 'Process PDF' to start chatting!")
    
    # Display example questions
    st.markdown("### 📝 Example Questions You Can Ask:")
    st.markdown("""
    - What is this document about?
    - Summarize the main points
    - What are the key findings?
    - Explain [specific topic] mentioned in the document
    - List the important dates/numbers mentioned
    """)