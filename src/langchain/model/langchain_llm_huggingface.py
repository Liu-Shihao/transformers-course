import os
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

"""
https://python.langchain.com/en/latest/modules/models/llms/integrations/huggingface_hub.html
使用LangChain 集成Hugging face 模型
"""




# get a token: https://huggingface.co/docs/api-inference/quicktour#get-your-api-token
from getpass import getpass

HUGGINGFACEHUB_API_TOKEN = getpass()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN



repo_id = "google/flan-t5-xl" # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who won the FIFA World Cup in the year 1994? "

print(llm_chain.run(question))