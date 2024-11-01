# Step 1: Environment Setup
import os
import time
import logging
import asyncio
from dotenv import load_dotenv
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the Groq API key as an environment variable
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# Import necessary modules from LangChain and related libraries
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# Load the Groq API key from the environment
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    logger.error("Groq API key not found in environment variables")
    raise ValueError("Groq API key is missing")

# Initialize the ChatGroq model with the API key and model name
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Step 2: Load and Prepare Interview Data
def load_and_prepare_data(file_path):
    try:
        data = pd.read_csv(file_path)
        questions = data['Question'].tolist()
        answers = data['Answer'].tolist()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = [chunk for question in questions for chunk in text_splitter.split_text(question)]
        
        logger.info(f"Loaded and prepared {len(chunks)} text chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error loading and preparing data: {str(e)}")
        raise

# Step 3: Generate Vector Embeddings
def generate_embeddings(text_chunks):
    try:
        embedding_model = OllamaEmbeddings(model="llama3.2")
        embeddings = [embedding_model.embed(chunk) for chunk in text_chunks]
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

# Step 4: Create FAISS Vector Store
def create_vector_store(chunks, embeddings):
    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        logger.info("Created FAISS vector store")
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise

# Step 5: Create RAG Chain with New Prompt
def create_rag_chain(llm, vectorstore):
    prompt = ChatPromptTemplate.from_template(
        """
        For the following interview question, provide a comprehensive response based on the context provided:

        Question: {input}

        Context:
        {context}

        Please structure your response as follows:
        1. Explain what the question means in simple terms.
        2. Describe what the interviewer expects in a response.
        3. Explain why this question is often asked in interviews.
        4. Describe the best approach to respond to this question (e.g., STAR method, CAR, CARE, or an appropriate logical sequence).
        5. Provide a clear explanation of this approach.
        6. Give a sample answer formatted as a brief, narrative response (not in bullet points).
        7. Highlight key takeaways and essential skills demonstrated in the response in one sentence.

        Ensure your response is clear, concise, and tailored to the specific question asked.
        """
    )
    
    retrieval_chain = create_retrieval_chain(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        document_prompt=prompt
    )
    logger.info("Created RAG chain with new prompt")
    return retrieval_chain

# Step 6: Query Function
async def get_answer(query, retrieval_chain):
    try:
        response = await asyncio.to_thread(retrieval_chain.run, {"question": query})
        logger.info(f"Generated response for query: {query}")
        return response
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise

# Step 7: FastAPI Setup
app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/get_answer")
async def get_answer_endpoint(query: Query):
    try:
        response = await get_answer(query.question, retrieval_chain)
        return {"answer": response}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request")

# Step 8: Main Execution
if __name__ == "__main__":
    try:
        # Load and prepare data
        chunks = load_and_prepare_data("interview_questions.csv")
        
        # Generate embeddings
        embeddings = generate_embeddings(chunks)
        
        # Create vector store
        vectorstore = create_vector_store(chunks, embeddings)
        
        # Create RAG chain with new prompt
        retrieval_chain = create_rag_chain(llm, vectorstore)
        
        # Run FastAPI
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.critical(f"Critical error in main execution: {str(e)}")
        raise