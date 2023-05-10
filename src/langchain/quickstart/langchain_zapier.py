from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
import os
os.environ["ZAPIER_NLA_API_KEY"] = ''
"""
结合使用 zapier 来实现将万种工具连接起来。
需要申请账号和他的自然语言 api key。https://zapier.com/l/natural-language-actions
他的 api key 虽然需要填写信息申请。但是基本填入信息后，基本可以秒在邮箱里看到审核通过的邮件。

然后，我们通过右键里面的连接打开我们的api 配置页面。我们点击右侧的 Manage Actions 来配置我们要使用哪些应用。

我们可以看到他成功读取了******@qq.com给他发送的最后一封邮件，并将总结的内容又发送给了******@qq.com
这只是个小例子，因为 zapier 有数以千计的应用，所以我们可以轻松结合 openai api 搭建自己的工作流。


"""
llm = OpenAI(temperature=.3)
zapier = ZapierNLAWrapper()
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
agent = initialize_agent(toolkit.get_tools(), llm, agent="zero-shot-react-description", verbose=True)

# 我们可以通过打印的方式看到我们都在 Zapier 里面配置了哪些可以用的工具
for tool in toolkit.get_tools():
  print (tool.name)
  print (tool.description)
  print ("\n\n")

agent.run('请用中文总结最后一封"******@qq.com"发给我的邮件。并将总结发送给"******@qq.com"')