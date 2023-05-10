from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, Document
import os
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

"""
https://gpt-index.readthedocs.io/en/latest/guides/primer/usage_pattern.html
"""
# 1. Load in Documents
documents = SimpleDirectoryReader('data').load_data()
index = GPTVectorStoreIndex.from_documents(documents)
# 加载文档：方式2
# text_list = ["text1", "text2", ...]
# documents = [Document(t) for t in text_list]

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)