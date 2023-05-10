import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
'''
https://huggingface.co/learn/nlp-course/zh-CN/chapter2/5?fw=pt
处理多个序列

标记化器的工作原理，并研究了标记化、到输入ID的转换、填充、截断和注意掩码。
'''
model_name = "/Users/liushihao/PycharmProjects/model-hub/distilbert-base-uncased-finetuned-sst-2-english"
checkpoint = model_name
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)


# This line will fail. RuntimeError: The size of tensor a (14) must match the size of tensor b (512) at non-singleton dimension 1
# input_ids = torch.tensor(ids)
# model(input_ids)

input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)

tokenized_inputs = tokenizer(sequence, return_tensors="pt")
print(tokenized_inputs["input_ids"])