import numpy as np
import gradio as gr
'''
图像示例
Gradio支持多种类型的组件，如 Image、DateFrame、Video或Label 。让我们尝试一个图像到图像的函数来感受一下！

当使用Image组件作为输入时，您的函数将接收一个形状为 (width, height, 3) 的NumPy数组，其中最后一个维度表示RGB值。我们还将以NumPy数组的形式返回一张图像。

你也可以用 type= 关键字参数设置组件使用的数据类型。例如，如果你想让你的函数获取一个图像的文件路径，而不是一个NumPy数组时，输入 Image 组件可以写成：
gr.Image(type="filepath", shape=...)
'''
def sepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img

demo = gr.Interface(sepia, gr.Image(shape=(200, 200)), "image")
demo.launch()