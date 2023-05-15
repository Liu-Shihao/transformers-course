from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
"""
https://huggingface.co/bigscience/bloom-1b7

currently only ('text2text-generation', 'text-generation') are supported
"""
model_path = "/Users/liushihao/PycharmProjects/model-hub/bigscience/bloom-1b7"

llm = HuggingFacePipeline.from_model_id(model_id=model_path, task="text-generation", model_kwargs={"temperature":0, "max_length":64})


template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

# question = "What is electroencephalography?"
question = "Who won the FIFA World Cup in the year 1994? "
print(llm_chain.run(question))