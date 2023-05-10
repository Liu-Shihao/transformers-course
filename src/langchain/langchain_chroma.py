from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader

# Load documents
# Load documents to do question answering over. If you want to do this over your documents, this is the section you should replace.
loader = TextLoader('state_of_the_union.txt')
documents = loader.load()

# Split documents
# Split documents into small chunks. This is so we can find the most relevant chunks for a query and pass only those into the LLM.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Initialize ChromaDB
# Create embeddings for each chunk and insert into the Chroma vector database.
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(texts, embeddings)

# Create the chain
# Initialize the chain we will use for question answering.
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb)

# Now we can use the chain to ask questions!
query = "What did the president say about Ketanji Brown Jackson"
qa.run(query)
