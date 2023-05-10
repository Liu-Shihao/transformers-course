import chromadb

#获取Chroma客户端
client = chromadb.Client()

#集合是您存储嵌入、文档和任何其他元数据的地方。您可以创建一个具有名称的集合：
collection = client.create_collection("sample_collection")

# 添加文档，Chroma 将存储您的文本，并自动处理标记化、嵌入和索引。
# Add docs to the collection. Can also update and delete. Row-based API coming soon!
collection.add(
    documents=["This is document1", "This is document2"], # we embed for you, or bring your own
    metadatas=[{"source": "notion"}, {"source": "google-docs"}], # filter on arbitrary metadata!
    ids=["doc1", "doc2"], # must be unique for each doc
)

# 如果您已经自己生成了嵌入，则可以直接将它们加载到：
collection.add(
    embeddings=[[1.2, 2.3, 4.5], [6.7, 8.2, 9.2]],
    documents=["This is a document", "This is another document"],
    metadatas=[{"source": "my_source"}, {"source": "my_source"}],
    ids=["id1", "id2"]
)

#查询
# 您可以使用查询文本列表来查询集合，Chroma 将返回最n相似的结果。就这么简单！
results = collection.query(
    query_texts=["This is a query document"],
    n_results=2,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)
print(results)