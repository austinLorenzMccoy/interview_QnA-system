# Interview Question Answering System

## Description

This project is an AI-powered interview question answering system built with FastAPI, LangChain, and Groq. It uses a Retrieval-Augmented Generation (RAG) approach to provide comprehensive and structured responses to interview questions.

The system loads a set of interview questions and answers, generates embeddings, and uses a FAISS vector store for efficient retrieval. When queried, it provides detailed explanations, context, and sample answers for interview questions.

## Features

- Load and process interview questions from a CSV file
- Generate vector embeddings for efficient retrieval
- Use RAG with Groq LLM for generating responses
- Provide structured answers including:
  - Simple explanation of the question
  - Interviewer expectations
  - Reason for asking the question
  - Best approach to answer (e.g., STAR method)
  - Sample answer
  - Key takeaways and essential skills demonstrated

## Prerequisites

- Python 3.8+
- Groq API key

## Installation

1. Clone the repository:
```

git clone [https://github.com/your-username/interview-qa-system.git](https://github.com/your-username/interview-qa-system.git)
cd interview-qa-system

```plaintext

2. Create a virtual environment and activate it:
```

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

```plaintext

3. Install the required packages:
```

pip install -r requirements.txt

```plaintext

4. Create a `.env` file in the project root and add your Groq API key:
```

GROQ_API_KEY=your_groq_api_key_here

```plaintext

## Usage

1. Prepare your interview questions CSV file:
- The file should be named `interview_questions.csv`
- It should have at least two columns: 'Question' and 'Answer'

2. Run the FastAPI application:
```

uvicorn app:app --reload

```plaintext

3. The API will be available at `http://127.0.0.1:8000`

4. To get an answer for an interview question, send a POST request to `/get_answer` endpoint with a JSON body:
```json
{
  "question": "Your interview question here"
}
```

## API Endpoints

- `POST /get_answer`: Get a comprehensive answer for an interview question


## Project Structure

- `app.py`: Main application file containing the FastAPI app and RAG system implementation
- `requirements.txt`: List of Python dependencies
- `.env`: Environment variables file (not tracked in git)
- `interview_questions.csv`: CSV file containing interview questions and answers
- `README.md`: This file, containing project documentation


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://python.langchain.com/)
- [Groq](https://groq.com/)
- [FAISS](https://github.com/facebookresearch/faiss)


```plaintext

This README.md file provides a comprehensive overview of your project, including:

1. A description of what the project does
2. Key features
3. Prerequisites
4. Installation instructions
5. Usage guide
6. API endpoint information
7. Project structure
8. Information for contributors
9. License information
10. Acknowledgements of key libraries and tools used
