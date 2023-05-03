from transformers import pipeline
'''
https://huggingface.co/facebook/bart-large-mnli
零样本分类
对尚未标记的文本进行分类
'''
classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

print(result)