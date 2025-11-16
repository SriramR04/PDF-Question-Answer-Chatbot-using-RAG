from groq import Groq
import os

# Initialize Groq client
GROQ_API_KEY = "  "
client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "llama-3.3-70b-versatile"

def generate_answer(question, context_chunks):
    """
    Generate answer using Groq's Llama model with RAG
    
    Args:
        question: User's question
        context_chunks: List of relevant text chunks from ChromaDB
        
    Returns:
        str: Generated answer
    """
    # Combine context chunks
    context = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
    
    # Create the prompt
    system_prompt = """You are a helpful AI assistant that answers questions based on the provided context from a PDF document. 

Instructions:
- Answer the question using ONLY the information provided in the context
- Be concise, clear, and accurate
- If the context doesn't contain enough information to answer the question, say so
- Don't make up information that's not in the context
- Use a professional and friendly tone"""

    user_prompt = f"""Context from PDF:
{context}

Question: {question}

Please provide a clear and accurate answer based on the context above."""

    try:
        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=MODEL_NAME,
            temperature=0.3,  # Lower temperature for more focused answers
            max_tokens=1024,
            top_p=0.9
        )
        
        answer = chat_completion.choices[0].message.content
        return answer
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"