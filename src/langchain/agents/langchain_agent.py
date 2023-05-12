from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import AgentType
import os
os.environ["OPENAI_API_KEY"] = 'OPENAI_API_KEY'
os.environ["SERPAPI_API_KEY"] = 'SERPAPI_API_KEY'
"""
pip install google-search-results
Serpapi 提供了 google 搜索的 api 接口。
需要到 Serpapi 官网上注册一个用户，https://serpapi.com/ 并复制他给我们生成 api key。
"""
# 加载 OpenAI 模型
llm = OpenAI(temperature=0,max_tokens=2048)

# 加载 serpapi 工具
tools = load_tools(["serpapi"])

# 如果搜索完想在计算一下可以这么写
# tools = load_tools(['serpapi', 'llm-math'], llm=llm)

# 如果搜索完想再让他再用python的print做点简单的计算，可以这样写
# tools=load_tools(["serpapi","python_repl"])

# 工具加载后都需要初始化，verbose 参数为 True，会打印全部的执行详情
# 在 chain 和 agent 对象上都会有 verbose 这个参数，这个是个非常有用的参数，开启他后我们可以看到完整的 chain 执行过程。
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
'''
AgentType
1. zero-shot-react-description: 根据工具的描述和请求内容的来决定使用哪个工具（最常用）
2. react-docstore: 使用 ReAct 框架和 docstore 交互, 使用Search 和Lookup 工具, 前者用来搜, 后者寻找term, 举例: Wipipedia 工具
3. self-ask-with-search 此代理只使用一个工具: Intermediate Answer, 它会为问题寻找事实答案(指的非 gpt 生成的答案, 而是在网络中,文本中已存在的), 如 Google search API 工具
4. conversational-react-description: 为会话设置而设计的代理, 它的prompt会被设计的具有会话性, 且还是会使用 ReAct 框架来决定使用来个工具, 并且将过往的会话交互存入内存
'''

# 运行 agent
agent.run("What's the date today? What great events have taken place today in history?")