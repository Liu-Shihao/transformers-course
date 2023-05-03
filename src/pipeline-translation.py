from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "/Users/liushihao/PycharmProjects/model-hub/Helsinki-NLP/opus-mt-zh-en"


model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

translator = pipeline("translation",
                      model=model,
                      tokenizer=tokenizer)

result = translator("你好，我是张三，很高兴认识你。")
print(result)