from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader

# 1.加载文档
# Load and process the text
loader = TextLoader('state_of_the_union.txt')
documents = loader.load()

# 2.分割文本
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 3.初始化一个持久化ChromaDB
# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
# 4.从磁盘加载数据，并初始化chain
# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb)

# 5，开始提问
query = "What did the president say about Ketanji Brown Jackson"
qa.run(query)
# 6.清除db
# To cleanup, you can delete the collection
vectordb.delete_collection()
vectordb.persist()