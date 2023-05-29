import gradio as gr

from src.api.huggingface_llm import ask_llm

'''

'''


demo = gr.Interface(
    fn=ask_llm,
    inputs=gr.Textbox(lines=2, placeholder="Search anything..."),
    outputs="text"
)
demo.launch()