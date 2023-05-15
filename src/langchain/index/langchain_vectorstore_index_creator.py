from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
"""
VectorstoreIndexCreator的内部工作流程
"""
# 加载文档
loader = TextLoader('../../example_data/state_of_the_union.txt', encoding='utf8')
documents = loader.load()

# 文本分割成块（chunk）
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 选择 embeddings
embeddings = OpenAIEmbeddings()

# 使用Chroma当做 vectorstore 矢量数据库
db = Chroma.from_documents(texts, embeddings)

# 在检索器接口中公开该索引
retriever = db.as_retriever()

# 创建一个链并用它来回答问题
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

query = "What did the president say about Ketanji Brown Jackson"
print(qa.run(query))
'''
" The President said that Judge Ketanji Brown Jackson is one of the nation's top legal minds, a former top litigator in private practice, a former federal public defender, and from a family of public school educators and police officers. He said she is a consensus builder and has received a broad range of support from organizations such as the Fraternal Order of Police and former judges appointed by Democrats and Republicans."
'''

# VectorstoreIndexCreator只是所有这些逻辑的包装。
# 它可以在它使用的文本拆分器、它使用的嵌入以及它使用的向量存储中进行配置。
# 例如，您可以配置如下：
index_creator = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=OpenAIEmbeddings(),
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
)