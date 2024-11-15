import gradio as gr
import requests
import json

# FastAPI server URL
API_URL = "http://127.0.0.1:8000"

# Define function to ask a question using FastAPI's /ask endpoint
def ask_question(question):
    response = requests.post(f"{API_URL}/ask", json={"question": question})
    if response.status_code == 200:
        return response.json().get("answer", "No response received.")
    else:
        return f"Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}"

# Start a conversation by calling the /conversation/start endpoint
def start_conversation():
    response = requests.post(f"{API_URL}/conversation/start")
    if response.status_code == 200:
        return response.json().get("status", "Failed to start conversation.")
    else:
        return f"Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}"

# Send a message in the conversation using the /conversation/message endpoint
def send_message(role, content):
    response = requests.post(f"{API_URL}/conversation/message", json={"role": role, "content": content})
    if response.status_code == 200:
        return response.json().get("status", "Failed to send message.")
    else:
        return f"Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}"

# Stop the conversation by calling the /conversation/stop endpoint
def stop_conversation():
    response = requests.post(f"{API_URL}/conversation/stop")
    if response.status_code == 200:
        return response.json().get("status", "Failed to stop conversation.")
    else:
        return f"Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}"

# Gradio interface setup
with gr.Blocks() as gradio_app:
    gr.Markdown("# Interview Q&A Bot")

    # Section for asking questions directly
    with gr.Tab("Ask a Question"):
        question_input = gr.Textbox(label="Enter your interview question:")
        answer_output = gr.Textbox(label="Answer from Interview Q&A Bot", interactive=False)
        ask_button = gr.Button("Ask")
        ask_button.click(fn=ask_question, inputs=question_input, outputs=answer_output)

    # Section for conversation mode
    with gr.Tab("Conversation Mode"):
        start_button = gr.Button("Start Conversation")
        conversation_output = gr.Textbox(label="Conversation Status", interactive=False)
        start_button.click(fn=start_conversation, outputs=conversation_output)

        role_input = gr.Radio(["user", "assistant"], label="Role", value="user")
        message_input = gr.Textbox(label="Enter your message:")
        message_status_output = gr.Textbox(label="Message Status", interactive=False)
        send_button = gr.Button("Send Message")
        send_button.click(fn=send_message, inputs=[role_input, message_input], outputs=message_status_output)

        stop_button = gr.Button("Stop Conversation")
        stop_button.click(fn=stop_conversation, outputs=conversation_output)

# Launch the Gradio app
if __name__ == "__main__":
    gradio_app.launch(share=True)
    
