from transformers import BertTokenizer
'''
https://huggingface.co/learn/nlp-course/zh-CN/chapter2/4?fw=pt
Tokenizer 分词器
标记器(Tokenizer)是 NLP 管道的核心组件之一。它们有一个目的：将文本转换为模型可以处理的数据。模型只能处理数字，因此标记器(Tokenizer)需要将我们的文本输入转换为数字数据。

基于词的(Word-based)
基于字符(Character-based)
为了两全其美，我们可以使用结合这两种方法的第三种技术：子词标记化(subword tokenization)。
子词分词算法依赖于这样一个原则，即不应将常用词拆分为更小的子词，而应将稀有词分解为有意义的子词。

加载和保存标记器(tokenizer)就像使用模型一样简单。实际上，它基于相同的两种方法： from_pretrained() 和 save_pretrained() 。
'''
model_name = "/Users/liushihao/PycharmProjects/model-hub/bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)

sequence = "Using a Transformer network is simple"

result = tokenizer(sequence)
print(result)
'''
{
'input_ids': [101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]
}
'''
# 编码 将文本翻译成数字被称为编码(encoding).编码分两步完成：标记化，然后转换为输入 ID。

tokens = tokenizer.tokenize(sequence)
print(tokens) #此方法的输出是一个字符串列表或标记(token)： ['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']

ids = tokenizer.convert_tokens_to_ids(tokens) #转换为适当的框架张量，就可以用作模型的输入 [7993, 170, 13809, 23763, 2443, 1110, 3014]
print(ids)

#解码 解码(Decoding) 正好相反：从词汇索引中，我们想要得到一个字符串。这可以通过 decode() 方法实现，如下：
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
'''
请注意， decode 方法不仅将索引转换回标记(token)，还将属于相同单词的标记(token)组合在一起以生成可读的句子。当我们使用预测新文本的模型（根据提示生成的文本，或序列到序列问题（如翻译或摘要））时，这种行为将非常有用。

到现在为止，您应该了解标记器(tokenizer)可以处理的原子操作：标记化、转换为 ID 以及将 ID 转换回字符串。
'''