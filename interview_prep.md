# Interview Coach API

This repository hosts the code for an AI-powered interview coach designed to help users practice interview questions in real time. The system is built with FastAPI, Groq's LLM API, and LangChain for Retrieval-Augmented Generation (RAG) capabilities, offering both HTTP and WebSocket-based interactions.

## Key Features

### 1. Model Selection and Integration
- **Model:** Groq's hosted Llama3-8b-8192 model
  - **Benefits:** Reduces computational needs, simplifies deployment, and enables faster inference.
- **Integration:** Utilizes `langchain_groq` for seamless compatibility with LangChain.

### 2. RAG System Implementation
- **Data:** Embedded knowledge base with sample interview questions and responses.
- **Components:**
  - **Text Splitter:** RecursiveCharacterTextSplitter for chunking
  - **Embeddings:** HuggingFace's "all-MiniLM-L6-v2"
  - **Vector Store:** FAISS for efficient similarity search
  - **Prompt Template:** Custom template designed for providing contextual responses.

### 3. Conversation Management
- **Stateful Tracking:** 
  - UUID-based conversation IDs.
  - Message history stored with roles for context-based responses.
  - Automatic session timeouts and configurable context windows.

### 4. FastAPI Endpoints
- **HTTP Endpoint (`/ask-question/`):** Handles traditional request-response interactions with conversation continuity.
- **WebSocket Endpoint (`/ws/chat/{conversation_id}`):** Supports real-time interactions with persistent connections and graceful exit management.

### 5. FastAPI Features Utilized
- Built-in request validation with Pydantic models.
- Asynchronous operations for non-blocking I/O.
- Swagger UI documentation available at `/docs`.

## Technical Details

### Project Structure
The code is organized into several core components:
- **`InterviewCoachConfig`:** Handles configuration and logging setup.
- **`InterviewCoach`:** Manages the application logic and RAG system.
- **Data Models:** Define data structures, including `Message`, `ConversationState`, and `QuestionInput`.

### Core Implementation

#### Environment Setup
```bash
conda create -n interview_coach python=3.10
conda activate interview_coach
pip install -r requirements.txt
```

#### Environment Variables
Ensure these are set in your `.env` file:
```
GROQ_API_KEY=your_api_key_here
TOKENIZERS_PARALLELISM=false
```

#### RAG System Setup
```python
def initialize_rag(self):
    chunks = self.process_csv_data(csv_data)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    self.vectorstore = FAISS.from_texts(chunks, embeddings)
    self.retrieval_chain = self.create_rag_chain()
```

#### Response Generation
```python
async def generate_response(self, question: str, conversation_id: str) -> str:
    # Get context, generate response, and update conversation state.
```

### Deployment
To run the application locally:
```bash
uvicorn interview_prep:app --host 0.0.0.0 --port 8000 --workers 4
```

## Optimization and Performance

- **Memory Management:** Conversation cleanup for inactive sessions.
- **Performance:** Async operations, optimized FAISS search, and structured prompt templates.
- **Error Handling:** Comprehensive logging, built-in FastAPI exception handling, and robust graceful degradation strategies.

---

**Note:** For a full overview of usage and customization, refer to the provided code and comments.

---

This documentation provides insights into the architecture, setup, and technical components that make this interview coaching API a robust and scalable solution for real-time conversational AI.