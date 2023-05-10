# LlamaIndex
llama-index的前身是gpt-index项目，最近才刚改名为'llama-index'。
LlamaIndex (GPT Index) 是一个提供中央接口以将您的 LLM 与外部数据连接起来的项目。
默认使用的是OpenAI GPT-3 `text-davinci-003` 模型，需要 `OPENAI_API_KEY`。

文档：https://gpt-index.readthedocs.io/en/latest/
Github：https://github.com/jerryjliu/llama_index
https://pypi.org/project/llama-index/



# Installation
```shell
pip install llama-index
```
可以参考搜索引擎中“先检索再重排”的思路，针对文档问答设计“先检索再整合“的方案，整体思路如下：
1. 首先准备好文档，并整理为纯文本的格式。把每个文档切成若干个小的chunks
2. 调用文本转向量的接口，将每个chunk转为一个向量，并存入向量数据库
3. 文本转向量可以使用openai embedding（https://platform.openai.com/docs/guides/embeddings/what-are-embeddings）
也可以使用其他方案，如fasttext/simbert等
4. 当用户发来一个问题的时候，将问题同样转为向量，并检索向量数据库，得到相关性最高的一个或几个chunk
5. 将问题和chunk合并重写为一个新的请求发给openai api，可能的请求格式如下：
上述“先检索再整合的逻辑”已经封装在llama-index库中


LlamaIndex 的一般使用模式如下：
加载文档（手动或通过数据加载器）
将文档解析为节点
构建索引（来自节点或文档）
[可选，高级] 在其他索引之上构建索引
查询索引
```python
from llama_index import StorageContext, load_index_from_storage,GPTVectorStoreIndex, SimpleDirectoryReader,LLMPredictor,ServiceContext,PromptHelper
from langchain import OpenAI
#加载数据
documents = SimpleDirectoryReader('data').load_data()

# 定义LLM ,默认使用的是 OpenAI 的 text-davinci-003 模型
# define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

# define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

index = GPTVectorStoreIndex.from_documents(
    documents, service_context=service_context
)


#创建索引（默认） 
index = GPTVectorStoreIndex.from_documents(documents)

# 保存索引到磁盘
index.storage_context.persist()
index.storage_context.persist(persist_dir="<persist_dir>")


# 从磁盘加载索引
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

# 查询索引
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)

response = query_engine.query("Write an email to the user given their background information.")
print(response)



```
