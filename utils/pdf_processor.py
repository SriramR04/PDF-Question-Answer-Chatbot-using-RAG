from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF using PyPDF2
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text from the entire PDF
    """
    reader = PdfReader(pdf_path)
    full_text = []
    
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text = page.extract_text()
        
        if text.strip():
            full_text.append(f"--- Page {page_num + 1} ---\n{text}")
    
    return "\n\n".join(full_text)


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into chunks with overlap
    
    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        list: List of text chunks
    """
    if not text or len(text) == 0:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        # Get the chunk
        chunk = text[start:end]
        
        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk.strip())
        
        # Move start position with overlap
        start = end - overlap
        
        # Prevent infinite loop
        if start <= end - chunk_size:
            start = end
    
    return chunks