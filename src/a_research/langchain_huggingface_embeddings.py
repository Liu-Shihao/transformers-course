from langchain import  HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms.base import LLM
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
"""
https://python.langchain.com/en/latest/use_cases/question_answering.html
https://python.langchain.com/en/latest/modules/indexes/retrievers/examples/vectorstore-retriever.html
1. 加载文档
2. 转换向量
3. 查询
"""


loader = TextLoader('../example_data/zoom_info.txt', encoding='utf8')

# index = VectorstoreIndexCreator().from_loaders([loader])
# query = "What did the president say about Ketanji Brown Jackson"
# index.query_with_sources(query)

# 1.Document Loaders
documents = loader.load()
print(documents)
# 2.Text Splitters
text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 3.Vectorstores
embeddings = HuggingFaceEmbeddings()
db = Chroma.from_documents(texts, embeddings)
# db = Chroma.from_documents(texts, embeddings,persist_directory="./data")
# db.persist()

# 4.VectorStore Retriever
retriever = db.as_retriever(search_kwargs={"k": 2})

# docs = retriever.get_relevant_documents("what is oscar's zoom ID?")
# print(docs)
#
# print(len(docs))

# 引入llm
repo_id = "google/flan-t5-xl"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})



qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query = "what is oscar's zoom ID?"
qa.run(query)