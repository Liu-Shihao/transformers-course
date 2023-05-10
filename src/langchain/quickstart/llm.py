from langchain.llms import OpenAI
import os
os.environ["OPENAI_API_KEY"] = "OpenAI-API-KEY"
llm = OpenAI(temperature=0.9)

text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))