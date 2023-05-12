from langchain.prompts import PromptTemplate
"""
PromptTemplate：管理 LLM 的prompt
"""
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

print(prompt.format(product="colorful socks"))
# What is a good name for a company that makes colorful socks?
