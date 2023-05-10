from transformers import pipeline
'''
Named entity recognition
命名实体识别 (NER) pipeline 负责从文本中抽取出指定类型的实体，例如人物、地点、组织等等。
'''
ner = pipeline("ner",
               grouped_entities=True,
               model="/Users/liushihao/PycharmProjects/model-hub/dbmdz/bert-large-cased-finetuned-conll03-english")
result = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(result)
'''
[{'entity_group': 'PER', 'score': 0.9981694, 'word': 'Sylvain', 'start': 11, 'end': 18}, 
{'entity_group': 'ORG', 'score': 0.9796021, 'word': 'Hugging Face', 'start': 33, 'end': 45}, 
{'entity_group': 'LOC', 'score': 0.9932106, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
在这里，模型正确地识别出 Sylvain 是一个人 (PER)，Hugging Face 是一个组织 (ORG)，而布鲁克林是一个位置 (LOC)。
'''