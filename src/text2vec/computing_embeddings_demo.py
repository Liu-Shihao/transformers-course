import sys

sys.path.append('..')
from text2vec import SentenceModel
from text2vec import Word2Vec
'''
基于pretrained model计算文本向量：
'''

def compute_emb(model):
    # Embed a list of sentences
    sentences = [
        '卡',
        '银行卡',
        '如何更换花呗绑定银行卡',
        '花呗更改绑定银行卡',
        'This framework generates embeddings for each input sentence',
        'Sentences are passed as a list of string.',
        'The quick brown fox jumps over the lazy dog.'
    ]
    sentence_embeddings = model.encode(sentences)
    print(type(sentence_embeddings), sentence_embeddings.shape)

    # The result is a list of sentence embeddings as numpy arrays
    for sentence, embedding in zip(sentences, sentence_embeddings):
        print("Sentence:", sentence)
        print("Embedding shape:", embedding.shape)
        print("Embedding head:", embedding[:10])
        print()


if __name__ == "__main__":
    model_name = "/Users/liushihao/PycharmProjects/model-hub/shibing624/text2vec-base-chinese"
    # 中文句向量模型(CoSENT)，中文语义匹配任务推荐，支持fine-tune继续训练
    t2v_model = SentenceModel(model_name)
    compute_emb(t2v_model)

    # 支持多语言的句向量模型（Sentence-BERT），英文语义匹配任务推荐，支持fine-tune继续训练
    # sbert_model = SentenceModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # compute_emb(sbert_model)

    # 中文词向量模型(word2vec)，中文字面匹配任务和冷启动适用
    # w2v_model = Word2Vec("w2v-light-tencent-chinese")
    # compute_emb(w2v_model)
'''
Sentence: 卡
Embedding shape: (768,)
Embedding head: [-0.35981628 -0.31868556  0.13830245  0.64885277 -0.01408509 -0.48738503
  1.3022574  -0.39590526  0.18606913  0.26202223]

Sentence: 银行卡
Embedding shape: (768,)
Embedding head: [ 0.8671564  -0.6067152   0.02293776  0.41320187 -0.5967977  -0.5103176
  1.4806962   0.85512793  0.22801666  0.63256425]

Sentence: 如何更换花呗绑定银行卡
Embedding shape: (768,)
Embedding head: [-4.4358693e-04 -2.9734713e-01  8.5790151e-01  6.9065183e-01
  3.9646000e-01 -8.4892666e-01 -1.9156845e-01  8.4548593e-02
  4.0232944e-01  3.1966126e-01]

Sentence: 花呗更改绑定银行卡
Embedding shape: (768,)
Embedding head: [ 0.6536199  -0.07666656  0.9596236   1.2794427  -0.00143495 -1.0384399
  0.13855317 -0.93946934  0.3380241   0.15471946]

Sentence: This framework generates embeddings for each input sentence
Embedding shape: (768,)
Embedding head: [-0.07267462  0.13551235  0.8715127   0.32199916  0.04113377 -1.4039385
  1.0236025   0.4870913  -0.32605395 -0.08317834]

Sentence: Sentences are passed as a list of string.
Embedding shape: (768,)
Embedding head: [-0.06514543  0.0745579   0.22670096  1.1061304  -0.2717633  -1.697876
  0.2945384  -0.127225    0.09710427 -0.36989343]

Sentence: The quick brown fox jumps over the lazy dog.
Embedding shape: (768,)
Embedding head: [-0.1987591  -0.6707982  -0.07586669  0.46342102  0.9580121  -0.76826847
  0.02202692  1.2257735  -0.42764106 -0.24545954]

'''