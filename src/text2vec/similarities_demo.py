from similarities import Similarity
'''
文本相似度计算和文本匹配搜索任务，推荐使用 similarities库 ，兼容本项目release的 Word2vec、SBERT、Cosent类语义匹配模型，还支持字面维度相似度计算、匹配搜索算法，支持文本、图像。
安装： pip install -U similarities
'''
m = Similarity()
# 句子相似度计算：
r = m.similarity('如何更换花呗绑定银行卡', '花呗更改绑定银行卡')
print(f"similarity score: {float(r)}")  # similarity score: 0.9476604461669922