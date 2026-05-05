import gradio as gr

def greet(message, history):
    return f"You said: {message}"

demo = gr.ChatInterface(fn=greet, title="Test")
demo.launch(server_name="0.0.0.0", server_port=7860)