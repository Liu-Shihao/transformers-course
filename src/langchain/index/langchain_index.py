from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

"""
VectorstoreIndexCreator 默认使用OpenAI 模型，需要设置OpenAI API KEY

"""

# 加载文档：使用TextLoader加载txt文件
loader = TextLoader('../state_of_the_union.txt', encoding='utf8')

# 创建索引：VectorstoreIndexCreator非常重要，直接将数据转换为向量
index = VectorstoreIndexCreator().from_loaders([loader])

# 可指定参数创建VectorstoreIndexCreator
index_creator = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=OpenAIEmbeddings(),
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
)
'''
加载文档后，在VectorstoreIndexCreator中将执行三个主要步骤：
1.将文档拆分成块 Splitting documents into chunks
2.为每个文档创建embeddings，Creating embeddings for each document
3.在 vectorstore 中存储文档和嵌入。Storing documents and embeddings in a vectorstore
'''

# 直接问问题
query = "What did the president say about Ketanji Brown Jackson"
index.query(query)
'''
" The president said that Ketanji Brown Jackson is one of the nation's top legal minds, a former top litigator in private practice, a former federal public defender, and from a family of public school educators and police officers. He also said that she is a consensus builder and has received a broad range of support from the Fraternal Order of Police to former judges appointed by Democrats and Republicans."
'''

# 响应中包含source 来源
query = "What did the president say about Ketanji Brown Jackson"
index.query_with_sources(query)
'''
{'question': 'What did the president say about Ketanji Brown Jackson',
 'answer': " The president said that he nominated Circuit Court of Appeals Judge Ketanji Brown Jackson, one of the nation's top legal minds, to continue Justice Breyer's legacy of excellence, and that she has received a broad range of support from the Fraternal Order of Police to former judges appointed by Democrats and Republicans.\n",
 'sources': '../state_of_the_union.txt'}
'''








