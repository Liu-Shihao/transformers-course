from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
text = "This is a test document."

query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text])