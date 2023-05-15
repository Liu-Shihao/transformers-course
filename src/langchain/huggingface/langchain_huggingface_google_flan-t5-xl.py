from langchain import PromptTemplate, LLMChain
from langchain import HuggingFaceHub
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

# 不支持本地模型
# Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/Users/liushihao/PycharmProjects/model-hub/google/flan-t5-xl'.  Use `repo_type` argument if needed.  (type=value_error.hfvalidation)

repo_id = "google/flan-t5-xl" # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who won the FIFA World Cup in the year 1994? "
# question = "谁赢得了1994年世界杯?"
print(llm_chain.run(question))