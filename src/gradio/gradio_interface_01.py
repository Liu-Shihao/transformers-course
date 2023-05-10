import gradio as gr
'''
为了制作演示，我们创建了一个gradio.Interface. 此类Interface可以使用用户界面包装任何 Python 函数。
在上面的示例中，我们看到了一个简单的基于文本的函数，但该函数可以是任何东西，从音乐生成器到税收计算器再到预训练机器学习模型的预测函数。

核心Interface类使用三个必需参数进行初始化：

fn: 环绕 UI 的函数
inputs: 哪个组件用于输入（例如"text","image"或"audio"）
outputs: 用于输出的组件（例如"text","image"或"label"）
'''

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()