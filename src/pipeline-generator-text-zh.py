from transformers import pipeline
'''
现在让我们看看如何使用pipeline来生成一些文本。这里的主要使用方法是您提供一个提示，模型将通过生成剩余的文本来自动完成整段话。
这类似于许多手机上的预测文本功能。文本生成涉及随机性，因此如果您没有得到相同的如下所示的结果，这是正常的。
使用参数 num_return_sequences 控制生成多少个不同的序列，并使用参数 max_length 控制输出文本的总长度
'''
generator = pipeline("text-generation",
                     model="/Users/liushihao/PycharmProjects/model-hub/bigscience/bloom-560m",
                     num_return_sequences=2,
                     max_length=30)
result = generator("你好，我打算五一去上海玩，请你给我做一个上海旅游攻略")
print(result)
