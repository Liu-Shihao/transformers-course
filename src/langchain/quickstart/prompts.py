from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

print(prompt.format(product="colorful socks"))
# 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
# export OPENAI_API_KEY="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"