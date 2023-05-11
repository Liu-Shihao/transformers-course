import gradio as gr
from text2vec import Similarity, EncoderType

"""
使用text2vec
计算文本相似度
"""



model_name = "/Users/liushihao/PycharmProjects/model-hub/shibing624/text2vec-base-chinese"

# 中文句向量模型(CoSENT)
sim_model = Similarity(model_name_or_path=model_name,
                       encoder_type=EncoderType.FIRST_LAST_AVG)


def ai_text(sentence1, sentence2):
    score = sim_model.get_score(sentence1, sentence2)
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentence1, sentence2, score))

    return score


if __name__ == '__main__':
    examples = [
        ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡'],
        ['我在北京打篮球', '我是北京人，我喜欢篮球'],
        ['一个女人在看书。', '一个女人在揉面团'],
        ['一个男人在车库里举重。', '一个人在举重。'],
    ]
    input1 = gr.inputs.Textbox(lines=2, placeholder="Enter First Sentence")
    input2 = gr.inputs.Textbox(lines=2, placeholder="Enter Second Sentence")

    output_text = gr.outputs.Textbox()
    gr.Interface(ai_text,
                 inputs=[input1, input2],
                 outputs=[output_text],
                 # theme="grass",
                 title="Chinese Text to Vector Model shibing624/text2vec-base-chinese",
                 description="Copy or input Chinese text here. Submit and the machine will calculate the cosine score.",
                 article="Link to <a href='https://github.com/shibing624/text2vec' style='color:blue;' target='_blank\'>Github REPO</a>",
                 examples=examples
                 ).launch()