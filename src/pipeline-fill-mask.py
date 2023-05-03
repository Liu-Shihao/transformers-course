from transformers import pipeline
'''

top_k 参数控制要显示的结果有多少种
这里模型填充了特殊的< mask >词，它通常被称为掩码标记。其他掩码填充模型可能有不同的掩码标记
'''
unmasker = pipeline("fill-mask")
result = unmasker("This course will teach you all about <mask> models.",
                  model="/Users/liushihao/PycharmProjects/model-hub/",
                  top_k=2)
print(result)