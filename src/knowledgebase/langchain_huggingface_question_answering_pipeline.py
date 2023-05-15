from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
import os

from langchain.document_loaders import TextLoader

os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
"""
question answering
https://huggingface.co/deepset/roberta-base-squad2
此方式不行：
ValueError: Got invalid task question-answering, currently only ('text2text-generation', 'text-generation') are supported
"""
# loader = TextLoader('../data/state_of_the_union.txt', encoding='utf8')
# documents = loader.load()
# print(documents)


model_path = "/Users/liushihao/PycharmProjects/model-hub/deepset/roberta-base-squad2"

# llm = HuggingFacePipeline.from_model_id(model_id=model_path, task="question-answering", model_kwargs={"temperature":0, "max_length":64})

template = """
    'question': {input_question},
    'context': {input_context}
"""
prompt = PromptTemplate.from_template(template)

# llm_chain = LLMChain(prompt=prompt, llm=llm)

# QA_input = {
#     'question': 'Why is model conversion important?',
#     'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
# }
print(prompt.format_messages(input_question="Why is model conversion important?",input_context="The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.")
)
# print(llm_chain.run(prompt.format_prompt(question="Why is model conversion important?",context="The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.")))