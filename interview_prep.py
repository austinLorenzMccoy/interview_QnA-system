#interview_prep.py

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, status
from pydantic import BaseModel, Field, validator
from groq import Groq
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Constants
INACTIVITY_TIMEOUT = 600  # 10 minutes
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MAX_TOKENS = 1024
TEMPERATURE = 0.5

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ConversationState(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    last_activity: datetime = Field(default_factory=datetime.now)
    active: bool = True

    class Config:
        arbitrary_types_allowed = True

class QuestionInput(BaseModel):
    question: str
    conversation_id: Optional[str] = None

    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

class InterviewCoachConfig:
    def __init__(self):
        load_dotenv()
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.exit_phrases = {"thank you", "thanks", "bye", "goodbye", "exit", "stop"}
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('interview_coach.log')
            ]
        )
        return logging.getLogger(__name__)

class InterviewCoach:
    def __init__(self):
        self.config = InterviewCoachConfig()
        self.logger = self.config.setup_logging()
        self.groq_client = Groq(api_key=self.config.groq_api_key)
        self.llm = ChatGroq(groq_api_key=self.config.groq_api_key, model_name="Llama3-8b-8192")
        self.conversation_storage: Dict[str, ConversationState] = {}
        
        # Initialize RAG components
        self.initialize_rag()

    def initialize_rag(self):
        try:
            # Embedded CSV data
            csv_data = [
                {"Question": "When you disagree with a coworker, how do you handle it?", "Answer": "There may be times when you and a co-worker disagree..."},
                {"Question": "Tell me about an instance when you had to juggle multiple tasks. How did you handle this situation?", "Answer": "As a software engineer, you may have several responsibilities to manage..."},
                {"Question": "Can you give me an example of how you establish your own goals?", "Answer": "Setting objectives is a crucial element of your job as a software engineer..."},
                {"Question": "Tell me about a moment when you were unfamiliar with the scenario or surroundings. How did you cope?", "Answer": "I had never worked as a full-time software developer before starting my last job..."},
                {"Question": "Tell me about a situation when you required information from someone unresponsive. How did you deal with it?", "Answer": "I was responsible for drafting a plan of action for my team..."},
                {"Question": "Tell me about a moment when you messed up. How did you correct your mistake?", "Answer": "In my previous position in the accounting business, I discovered that I had planned a meeting..."},
                {"Question": "Give me an example of how you've worked as part of a team.", "Answer": "At my previous job, I was a key member of our SEO team..."},
                {"Question": "What is the most helpful piece of feedback you've ever received about yourself?", "Answer": "My boss brought me into her office a year ago and gave me some critical comments..."},
                {"Question": "What is the best way to answer Behavioral interview questions?", "Answer": "Employers are searching for a detailed explanation of a previous experience..."}
            ]
            
            # Process and prepare the CSV data for use in the RAG system
            chunks = self.process_csv_data(csv_data)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vectorstore = FAISS.from_texts(chunks, embeddings)
            self.retrieval_chain = self.create_rag_chain()
            self.logger.info("RAG system initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG system: {str(e)}")
            raise

    def process_csv_data(self, csv_data: List[Dict[str, str]]) -> List[str]:
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = []
            for entry in csv_data:
                question = entry["Question"]
                answer = entry["Answer"]
                combined_text = f"Question: {question}\nAnswer: {answer}"
                chunks.extend(text_splitter.split_text(combined_text))
            self.logger.info(f"Processed {len(chunks)} text chunks from embedded CSV data")
            return chunks
        except Exception as e:
            self.logger.error(f"Error processing CSV data: {str(e)}")
            raise

    def create_rag_chain(self) -> RetrievalQA:
        prompt = ChatPromptTemplate.from_template("""
        You are an expert interview coach helping prepare candidates for interviews. 
        Use the following context and question to provide a helpful response. 
        If the context is relevant, incorporate it into your response. 
        If not, provide a new response based on best practices for interviews.

        Context: {context}
        Question: {question}

        Remember to:
        1. Use the STAR or CAR framework when applicable
        2. Be concise but specific
        3. Focus on actionable advice
        4. Include quantifiable results when possible

        Response:""")

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )

    async def generate_response(self, question: str, conversation_id: str) -> str:
        try:
            # Check if it's the first interaction in the conversation
            if conversation_id not in self.conversation_storage or len(self.conversation_storage[conversation_id].messages) == 0:
                # First message: Provide an introductory response
                intro_response = (
                    "Hello! I'm your interview coach. Feel free to ask me anything, "
                    "and I'll help you prepare for your upcoming interviews. "
                    "For your first question, I don't have any context yet, but I'll answer directly."
                )
                self.add_message_to_conversation(conversation_id, MessageRole.USER, question)
                self.add_message_to_conversation(conversation_id, MessageRole.ASSISTANT, intro_response)
                return intro_response

            # If there is context, proceed with generating a response based on it
            context = self.get_conversation_context(conversation_id)
            prompt = f"""Previous conversation context:
            {context}

            Current question: {question}

            Please provide a response that takes into account the conversation history if relevant.
            If there is no relevant context, provide a direct answer to the question."""
            
            chat_completion = await asyncio.to_thread(
                self.groq_client.chat.completions.create,
                messages=[{"role": "user", "content": prompt}],
                model="Llama3-8b-8192",
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            
            response = chat_completion.choices[0].message.content
            self.add_message_to_conversation(conversation_id, MessageRole.USER, question)
            self.add_message_to_conversation(conversation_id, MessageRole.ASSISTANT, response)
            
            return response
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate response"
            )

    def get_conversation_context(self, conversation_id: str, context_window: int = 5) -> str:
        if conversation_id not in self.conversation_storage:
            return ""
        
        conversation = self.conversation_storage[conversation_id]
        recent_messages = conversation.messages[-context_window:]
        return "\n".join([f"{msg.role}: {msg.content}" for msg in recent_messages])

    def add_message_to_conversation(self, conversation_id: str, role: MessageRole, content: str):
        if conversation_id not in self.conversation_storage:
            self.conversation_storage[conversation_id] = ConversationState()
        
        conversation = self.conversation_storage[conversation_id]
        conversation.messages.append(Message(role=role, content=content))
        conversation.last_activity = datetime.now()

    def clean_inactive_conversations(self):
        current_time = datetime.now()
        for conversation_id, conversation in list(self.conversation_storage.items()):
            if (current_time - conversation.last_activity) > timedelta(seconds=INACTIVITY_TIMEOUT):
                conversation.active = False
                self.logger.info(f"Conversation {conversation_id} marked as inactive due to timeout")

# FastAPI App Setup
app = FastAPI()

# Initialize InterviewCoach
interview_coach = InterviewCoach()

@app.post("/ask-question/")
async def ask_question(input: QuestionInput):
    conversation_id = input.conversation_id or str(uuid.uuid4())
    response = await interview_coach.generate_response(input.question, conversation_id)
    return {"conversation_id": conversation_id, "response": response}

@app.websocket("/ws/chat/{conversation_id}")
async def websocket_chat(websocket: WebSocket, conversation_id: str):
    await websocket.accept()
    await websocket.send_text(f"Welcome! Let's start preparing for your interview. Ask your first question.")
    try:
        while True:
            message = await websocket.receive_text()
            if any(exit_phrase in message.lower() for exit_phrase in interview_coach.config.exit_phrases):
                await websocket.send_text("Goodbye! Let me know if you'd like to continue later.")
                break

            response = await interview_coach.generate_response(message, conversation_id)
            await websocket.send_text(response)
    except WebSocketDisconnect:
        interview_coach.logger.info(f"Connection closed for conversation {conversation_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)