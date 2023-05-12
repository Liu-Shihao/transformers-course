from langchain import OpenAI, ConversationChain
"""
ConversationChain
"""

# 默认情况下，ConversationChain有一种简单类型的内存，可以记住所有以前的输入/输出并将它们添加到传递的上下文中。让我们来看看使用这个链（设置verbose=True以便我们可以看到提示）。
llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="Hi there!")
print(output)

output = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
print(output)