# gradio_app.py

import requests
import gradio as gr

# URL for the FastAPI endpoint
FASTAPI_URL = "http://localhost:8000/get_answer"

# Function to send question to FastAPI and retrieve answer
def get_answer_from_fastapi(question):
    try:
        response = requests.post(FASTAPI_URL, json={"question": question})
        if response.status_code == 200:
            answer = response.json().get("answer", "No answer available")
        else:
            answer = f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        answer = f"Error communicating with the server: {str(e)}"
    return answer

# Gradio Interface
interface = gr.Interface(
    fn=get_answer_from_fastapi,
    inputs="text",
    outputs="text",
    title="Interview Question Answering API",
    description="Ask an interview question, and get a structured answer."
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch(share=True)
