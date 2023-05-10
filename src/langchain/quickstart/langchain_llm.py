from langchain.llms import OpenAI
import os
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

# 使用LangChain加载LLM，默认是OpenAI，也可以自定义
llm = OpenAI(temperature=0.9)
llm = OpenAI(model_name="text-davinci-003",max_tokens=1024)
text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))