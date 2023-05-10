import sys
'''
句子相似度计算
'''
sys.path.append('..')
from text2vec import Similarity

# Two lists of sentences
sentences1 = ['如何更换花呗绑定银行卡',
              'The cat sits outside',
              'A man is playing guitar',
              'The new movie is awesome']

sentences2 = ['花呗更改绑定银行卡',
              'The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great']

sim_model = Similarity()
for i in range(len(sentences1)):
    for j in range(len(sentences2)):
        score = sim_model.get_score(sentences1[i], sentences2[j])
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[j], score))
'''
句子余弦相似度值score范围是[-1, 1]，值越大越相似。

如何更换花呗绑定银行卡 		 花呗更改绑定银行卡 		 Score: 0.9477
如何更换花呗绑定银行卡 		 The dog plays in the garden 		 Score: -0.1748
如何更换花呗绑定银行卡 		 A woman watches TV 		 Score: -0.0839
如何更换花呗绑定银行卡 		 The new movie is so great 		 Score: -0.0044
The cat sits outside 		 花呗更改绑定银行卡 		 Score: -0.0097
The cat sits outside 		 The dog plays in the garden 		 Score: 0.1908
The cat sits outside 		 A woman watches TV 		 Score: -0.0203
The cat sits outside 		 The new movie is so great 		 Score: 0.0302
A man is playing guitar 		 花呗更改绑定银行卡 		 Score: -0.0010
A man is playing guitar 		 The dog plays in the garden 		 Score: 0.1062
A man is playing guitar 		 A woman watches TV 		 Score: 0.0055
A man is playing guitar 		 The new movie is so great 		 Score: 0.0097
The new movie is awesome 		 花呗更改绑定银行卡 		 Score: 0.0302
The new movie is awesome 		 The dog plays in the garden 		 Score: -0.0160
The new movie is awesome 		 A woman watches TV 		 Score: 0.1321
The new movie is awesome 		 The new movie is so great 		 Score: 0.9591
'''