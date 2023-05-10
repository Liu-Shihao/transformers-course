import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

'''
https://huggingface.co/learn/nlp-course/zh-CN/chapter2/2?fw=pt
我们使用AutoTokenizer类及其from_pretrained()方法。使用我们模型的检查点名称，它将自动获取与模型的标记器相关联的数据，并对其进行缓存（因此只有在您第一次运行下面的代码时才会下载）。

因为sentiment-analysis（情绪分析）管道的默认检查点是distilbert-base-uncased-finetuned-sst-2-english

一旦我们有了标记器，我们就可以直接将我们的句子传递给它，然后我们就会得到一本字典，它可以提供给我们的模型！剩下要做的唯一一件事就是将输入ID列表转换为张量。
'''


checkpoint = "/Users/liushihao/PycharmProjects/model-hub/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModel.from_pretrained(checkpoint)
# 对于本示例，我们需要一个带有序列分类头的模型（能够将句子分类为肯定或否定）。因此，我们实际上不会使用AutoModel类，而是使用AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
'''
您可以使用🤗 Transformers，而不必担心哪个ML框架被用作后端；它可能是PyTorch或TensorFlow，或Flax。
但是，Transformers型号只接受张量作为输入。如果这是你第一次听说张量，你可以把它们想象成NumPy数组。NumPy数组可以是标量（0D）、向量（1D）、矩阵（2D）或具有更多维度。
它实际上是张量；其他ML框架的张量行为类似，通常与NumPy数组一样易于实例化。
以下是PyTorch张量的结果：
{'input_ids': tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
          2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,
             0,     0,     0,     0,     0,     0]]), 
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}
输出本身是一个包含两个键的字典，input_ids和attention_mask。input_ids包含两行整数（每个句子一行），它们是每个句子中标记的唯一标记（token）
'''

outputs = model(**inputs)
# print(outputs.last_hidden_state.shape) #torch.Size([2, 16, 768])
'''
torch.Size([2, 16, 768])
Transformers模块的矢量输出通常较大。它通常有三个维度：
Batch size: 一次处理的序列数（在我们的示例中为2）。
Sequence length: 序列的数值表示的长度（在我们的示例中为16）。
Hidden size: 每个模型输入的向量维度。
'''

print(outputs.logits.shape) # torch.Size([2, 2]) 因为我们只有两个句子和两个标签，所以我们从模型中得到的结果是2 x 2的形状。
print(outputs.logits)
'''
我们的模型预测第一句为[-1.5607, 1.6123]，第二句为[ 4.1692, -3.3464]。这些不是概率，而是logits，即模型最后一层输出的原始非标准化分数。要转换为概率，它们需要经过SoftMax层（所有🤗Transformers模型输出logits，因为用于训练的损耗函数通常会将最后的激活函数（如SoftMax）与实际损耗函数（如交叉熵）融合）：
'''

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
print(model.config.id2label)
'''
现在我们可以看到，模型预测第一句为[0.0402, 0.9598]，第二句为[0.9995, 0.0005]。这些是可识别的概率分数。
为了获得每个位置对应的标签，我们可以检查模型配置的id2label属性

现在我们可以得出结论，该模型预测了以下几点：

第一句：否定：0.0402，肯定：0.9598
第二句：否定：0.9995，肯定：0.0005
'''
