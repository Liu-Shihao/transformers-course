from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
"""
创建SystemMessagePromptTemplate和HumanMessagePromptTemplate
并根据system_message_prompt和human_message_prompt创建ChatPromptTemplate
然后将ChatPromptTemplate传递给LLM
"""
# chat = ChatOpenAI(temperature=0)

template = "You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
print(system_message_prompt.format_messages(input_language="Chinese",output_language="Englist"))

human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
print(human_message_prompt.format_messages(text="Hello LangChain!"))

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

print(chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.").to_messages())

# get a chat completion from the formatted messages
# chat(chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.").to_messages())
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})