from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''

"""
使用在线的 HuggingFace 模型
"""

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":64})
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
print(llm_chain.run(question))