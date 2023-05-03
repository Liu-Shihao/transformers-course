from transformers import pipeline
'''
https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
distilbert-base-uncased-finetuned-sst-2-english
默认情况下，此pipeline选择一个特定的预训练模型，该模型已针对英语情感分析进行了微调。创建分类器对象时，将下载并缓存模型。如果您重新运行该命令，则将使用缓存的模型，无需再次下载模型。

将一些文本传递到pipeline时涉及三个主要步骤：

1.文本被预处理为模型可以理解的格式。
2.预处理的输入被传递给模型。
3.模型处理后输出最终人类可以理解的结果。
'''
classifier = pipeline("sentiment-analysis")
result = classifier([
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ])

print(result)
'''
[{'label': 'POSITIVE', 'score': 0.9598050713539124}, 
{'label': 'NEGATIVE', 'score': 0.9994558691978455}]
'''