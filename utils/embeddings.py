from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os
from utils.pdf_processor import chunk_text

# Initialize the embedding model
MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
embedding_model = SentenceTransformer(MODEL_NAME)

# ChromaDB setup
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_documents"

def get_chroma_client():
    """Initialize and return ChromaDB client"""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client

def create_embeddings(text_content):
    """
    Create embeddings from text and store in ChromaDB
    
    Args:
        text_content: Full text extracted from PDF
    """
    # Chunk the text
    chunks = chunk_text(text_content, chunk_size=500, overlap=50)
    
    if not chunks:
        raise ValueError("No text chunks created from PDF")
    
    # Generate embeddings
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    
    # Initialize ChromaDB
    client = get_chroma_client()
    
    # Delete existing collection if it exists (for new PDF)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except:
        pass
    
    # Create new collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Using cosine similarity
    )
    
    # Prepare data for ChromaDB
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # Add documents to collection
    collection.add(
        embeddings=embeddings.tolist(),
        documents=chunks,
        ids=ids
    )
    
    print(f"Created {len(chunks)} chunks and stored embeddings in ChromaDB")

def query_embeddings(question, top_k=3):
    """
    Query ChromaDB for relevant chunks
    
    Args:
        question: User's question
        top_k: Number of top results to retrieve
        
    Returns:
        list: List of relevant text chunks
    """
    # Generate embedding for the question
    question_embedding = embedding_model.encode([question], show_progress_bar=False)
    
    # Query ChromaDB
    client = get_chroma_client()
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except:
        return []
    
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=top_k
    )
    
    # Extract documents from results
    if results and 'documents' in results and len(results['documents']) > 0:
        return results['documents'][0]  # Returns list of top_k documents
    
    return []