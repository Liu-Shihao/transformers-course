from transformers import AutoTokenizer
'''
https://huggingface.co/learn/nlp-course/zh-CN/chapter2/6?fw=pt
'''
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)