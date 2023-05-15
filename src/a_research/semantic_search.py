import sys
import gradio as gr

sys.path.append('..')
from text2vec import SentenceModel, cos_sim, semantic_search

"""
文本匹配搜索
一般在文档候选集中找与query最相似的文本，常用于QA场景的问句相似匹配、文本相似检索等任务。
"""
embedder = SentenceModel()

# Corpus with example sentences
corpus = [
    'An embedding is a vector (list) of floating point numbers. The distance between two vectors measures their relatedness. Small distances suggest high relatedness and large distances suggest low relatedness.',
    'We recommend cosine similarity. The choice of distance function typically doesn’t matter much.',
    'In Python, you can split a string into tokens with OpenAI\'s tokenizer tiktoken.',
    'For searching over many vectors quickly, we recommend using a vector database. '

]
corpus_embeddings = embedder.encode(corpus)

# Query sentences:
queries = [
    'What are embeddings?',
    'Which distance function should I use?',
    'How can I tell how many tokens a string has before I embed it?',
    'How can I retrieve K nearest embedding vectors quickly?',
]


def ask(query):
    answer = ''
    query_embedding = embedder.encode(query)
    hits = semantic_search(query_embedding, corpus_embeddings, top_k=5)

    answer += "\nTop 5 most similar sentences in corpus:"
    hits = hits[0]  # Get the hits for the first query
    for hit in hits:
        answer += "\n\n============================================\n\n"
        answer += (corpus[hit['corpus_id']] + "(Score: {:.4f})".format(hit['score']))
    return answer


if __name__ == '__main__':
    input = gr.inputs.Textbox(lines=2, placeholder="ask...")
    output_text = gr.outputs.Textbox()
    gr.Interface(ask,
                 inputs=[input],
                 outputs=[output_text],
                 # theme="grass",
                 title="You Know, For Search",
                 ).launch()
