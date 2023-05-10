import sys

sys.path.append('..')
from text2vec import SentenceModel, cos_sim, semantic_search
'''
文本匹配搜索
一般在文档候选集中找与query最相似的文本，常用于QA场景的问句相似匹配、文本相似检索等任务。
'''
embedder = SentenceModel()

# Corpus with example sentences
corpus = [
    '花呗更改绑定银行卡',
    '我什么时候开通了花呗',
    'A man is eating food.',
    'A man is eating a piece of bread.',
    'The girl is carrying a baby.',
    'A man is riding a horse.',
    'A woman is playing violin.',
    'Two men pushed carts through the woods.',
    'A man is riding a white horse on an enclosed ground.',
    'A monkey is playing drums.',
    'A cheetah is running behind its prey.'
]
corpus_embeddings = embedder.encode(corpus)

# Query sentences:
queries = [
    '如何更换花呗绑定银行卡',
    'A man is eating pasta.',
    'Someone in a gorilla costume is playing a set of drums.',
    'A cheetah chases prey on across a field.']

for query in queries:
    query_embedding = embedder.encode(query)
    hits = semantic_search(query_embedding, corpus_embeddings, top_k=5)
    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")
    hits = hits[0]  # Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))

'''


======================


Query: 如何更换花呗绑定银行卡

Top 5 most similar sentences in corpus:
花呗更改绑定银行卡 (Score: 0.8551)
我什么时候开通了花呗 (Score: 0.7212)
A man is eating food. (Score: 0.3118)
A man is eating a piece of bread. (Score: 0.2992)
A monkey is playing drums. (Score: 0.2922)


======================


Query: A man is eating pasta.

Top 5 most similar sentences in corpus:
A man is eating food. (Score: 0.7840)
A man is riding a white horse on an enclosed ground. (Score: 0.6906)
A man is eating a piece of bread. (Score: 0.6831)
A man is riding a horse. (Score: 0.6515)
Two men pushed carts through the woods. (Score: 0.5270)


======================


Query: Someone in a gorilla costume is playing a set of drums.

Top 5 most similar sentences in corpus:
A monkey is playing drums. (Score: 0.6758)
A man is riding a white horse on an enclosed ground. (Score: 0.6351)
The girl is carrying a baby. (Score: 0.5438)
A man is riding a horse. (Score: 0.5002)
A man is eating a piece of bread. (Score: 0.4916)


======================


Query: A cheetah chases prey on across a field.

Top 5 most similar sentences in corpus:
A cheetah is running behind its prey. (Score: 0.6736)
A man is riding a white horse on an enclosed ground. (Score: 0.5731)
A monkey is playing drums. (Score: 0.4977)
The girl is carrying a baby. (Score: 0.4570)
A man is riding a horse. (Score: 0.4287)
'''