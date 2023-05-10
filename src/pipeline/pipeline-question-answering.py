from transformers import pipeline
'''
自动问答
此pipeline通过从提供的上下文中提取信息来工作；它不会凭空生成答案。
'''
question_answerer = pipeline("question-answering")
question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)