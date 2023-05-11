from langchain.embeddings import HuggingFaceEmbeddings

"""
Let’s load the Hugging Face Embedding class.
"""

embeddings = HuggingFaceEmbeddings()
text = "This is a test document."
query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text])