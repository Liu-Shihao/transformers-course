from langchain.embeddings import HuggingFaceEmbeddings

"""
Let’s load the Hugging Face Embedding class.


"""

model_name = "/Users/liushihao/PycharmProjects/model-hub/shibing624/text2vec-base-chinese"

embeddings = HuggingFaceEmbeddings(model_name=model_name)
text = "This is a test document."
query_result = embeddings.embed_query(text)
doc_result = embeddings.embed_documents([text])