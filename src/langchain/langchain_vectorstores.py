from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI,VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
"""
构建本地知识库问答机器人
在这个例子中，我们会介绍如何从我们本地读取多个文档构建知识库，并且使用 Openai API 在知识库中进行搜索并给出答案。

这个是个很有用的教程，比如可以很方便的做一个可以介绍公司业务的机器人，或是介绍一个产品的机器人。

"""


# 加载文件夹中的所有txt类型的文件
loader = DirectoryLoader('/content/sample_data/data/', glob='**/*.txt')
# 将数据转成 document 对象，每个文件会作为一个 document
documents = loader.load()

# 初始化加载器
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# 切割加载的 document
split_docs = text_splitter.split_documents(documents)

# 初始化 openai 的 embeddings 对象
embeddings = OpenAIEmbeddings()
# 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
docsearch = Chroma.from_documents(split_docs, embeddings)

# 创建问答对象
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch,return_source_documents=True)
# 进行问答
result = qa({"query": "科大讯飞今年第一季度收入是多少？"})
print(result)

'''
将 document 信息转换成向量信息和embeddings的信息并临时存入 Chroma 数据库。
因为是临时存入，所以当我们上面的代码执行完成后，上面的向量化后的数据将会丢失。如果想下次使用，那么就还需要再计算一次embeddings

chroma 是个本地的向量数据库，他提供的一个 persist_directory 来设置持久化目录进行持久化。读取时，只需要调取 from_document 方法加载即可。
'''
# 持久化数据
docsearch = Chroma.from_documents(documents, embeddings, persist_directory="D:/vector_store")
docsearch.persist()

# 加载数据
docsearch = Chroma(persist_directory="D:/vector_store", embedding_function=embeddings)