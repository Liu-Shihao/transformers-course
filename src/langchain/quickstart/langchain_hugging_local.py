from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import os

os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''
"""
将 HuggingFace 模型直接拉到本地使用

将模型拉到本地使用的好处：

训练模型
可以使用本地的 GPU
有些模型无法在 HuggingFace 运行
"""


model_id = 'google/flan-t5-large'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=100
)

local_llm = HuggingFacePipeline(pipeline=pipe)
print(local_llm('What is the capital of France? '))

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt,  llm=local_llm)
question = "What is the capital of England?"
print(llm_chain.run(question))