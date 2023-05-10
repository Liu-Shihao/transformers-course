import chromadb
from llama_index.vector_stores import ChromaVectorStore
"""
Using Vector Stores
https://gpt-index.readthedocs.io/en/latest/how_to/integrations/vector_stores.html#vector-store-index
"""
# Creating a Chroma client
# By default, Chroma will operate purely in-memory.
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("quickstart")

# construct vector store
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection,
)