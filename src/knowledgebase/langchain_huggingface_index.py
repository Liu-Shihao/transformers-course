from langchain import HuggingFaceHub
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# get a token: https://huggingface.co/docs/api-inference/quicktour#get-your-api-token
from getpass import getpass

# HUGGINGFACEHUB_API_TOKEN = getpass()
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
"""
使用google/flan-t5-xl 文本生成 模型
将数据转化为向量，存储到Chroma VectorStore
"""
loader = TextLoader('../example_data/zoom_info.txt', encoding='utf8')
documents = loader.load()

repo_id = "google/flan-t5-xl"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})


# embeddings 默认：https://huggingface.co/sentence-transformers/all-mpnet-base-v2
index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=HuggingFaceEmbeddings(),
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
).from_loaders([loader])

query = "What is the oscar zoom info?"
index.query_with_sources(question=query,llm=llm)