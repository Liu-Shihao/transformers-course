from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex, PromptHelper, LLMPredictor, ServiceContext,GPTVectorStoreIndex
from langchain import OpenAI
import gradio as gr
import sys
import os
os.chdir(r'./data')
os.environ["OPENAI_API_KEY"] = 'OPENAI_API_KEY'

"""
需要OpenAI API KEY,使用了OpenAI的gpt-3.5-turbo模型

使用gradio构建WebUI
使用llama_index构建文本向量

调用construct_index函数，首先会将该目录下的所有文件转化为index，保存到磁盘
调用第二个函数chatbot，则从磁盘上加载index，并进行查询，返回响应
"""

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 2000
    max_chunk_overlap = 20
    chunk_size_limit = 600
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    documents = SimpleDirectoryReader(directory_path).load_data()
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    index = GPTVectorStoreIndex.from_documents(documents,service_context=service_context)
    index.save_to_disk('index.json')
    return index
def chatbot(input_text):
    index = GPTVectorStoreIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response
iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(lines=7, label="输入您的文本"),
                     outputs="text",
                     title="AI 知识库聊天机器人")
index = construct_index("docs")
iface.launch(share=True)