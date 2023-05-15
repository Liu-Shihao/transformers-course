from langchain.embeddings import HuggingFaceEmbeddings
"""
默认：https://huggingface.co/sentence-transformers/all-mpnet-base-v2
"""
embeddings = HuggingFaceEmbeddings()

text = "This is a test document."

query_result = embeddings.embed_query(text)
print(query_result)

doc_result = embeddings.embed_documents([text])
print(doc_result)