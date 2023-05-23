from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms.base import LLM
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
"""
https://python.langchain.com/en/latest/use_cases/question_answering.html
https://python.langchain.com/en/latest/modules/indexes/retrievers/examples/vectorstore-retriever.html
"""


loader = TextLoader('../example_data/test.txt', encoding='utf8')

# index = VectorstoreIndexCreator().from_loaders([loader])
# query = "What did the president say about Ketanji Brown Jackson"
# index.query_with_sources(query)

# 1.Document Loaders
documents = loader.load()
# print(documents)

# 2.Text Splitters
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# select embeddings
# 默认：sentence-transformers/all-mpnet-base-v2，中文：GanymedeNil/text2vec-large-chinese
embeddings = HuggingFaceEmbeddings()
# 3.create vectorstores
db = Chroma.from_documents(texts, embeddings)
# db = Chroma.from_documents(texts, embeddings,persist_directory="./data")
# db.persist()

# 4. Retriever
retriever = db.as_retriever(search_kwargs={"k": 2})

query = "what is embeddings?"
docs = retriever.get_relevant_documents(query)
# print(docs)
# print(len(docs))
for item in docs:
    print("page_content:")
    print(item.page_content)
    print("source:")
    print(item.metadata['source'])
    print("---------------------------")

# select llm:
# google/flan-t5-xl
# lmsys/fastchat-t5-3b-v1.0
# TheBloke/wizardLM-7B-HF
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15
)

llm = HuggingFacePipeline(pipeline=pipe)
# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# query = "what is embeddings?"
# llm_response = qa.run(query)
# print("LLM Answer:")
# print(llm_response)
# print("done.")

chain = load_qa_chain(llm, chain_type="stuff")
llm_response = chain.run(input_documents=docs, question=query)
print(llm_response)
print("done.")