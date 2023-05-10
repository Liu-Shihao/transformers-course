import gradio as gr
'''
Gradio 提供了两个类来构建应用程序 :Interface 和 Blocks
Blocks: 更加灵活且可控

1. Interface，这为创建到目前为止我们一直在讨论的示例提供了一个高级抽象。

2. Blocks，一个用于设计具有更灵活布局和数据流的web应用程序的初级API。block可以做许多事，比如特征化多个数据流和演示，控制组件在页面上出现的位置，处理复杂的数据流（例如，输出可以作为其他函数的输入），以及根据用户交互更新组件的属性/可见性，且仍然在Python中。如果您需要这种个性化，那就试试 Blocks 吧！
注意事项：

Blocks 由 with 子句组成，在该子句中创建的任何组件都会自动添加到应用程序中。
组件在应用程序中按创建的顺序垂直显示，（稍后我们将介绍自定义布局！）
一个 按钮 Button 被创建，然后添加了一个 click 事件监听器。这个API看起来很熟悉！就像 Interface一样， click 方法接受一个Python函数、输入组件和输出组件。
'''
def greet(name):
    return "Hello " + name + "!"

with gr.Blocks() as demo:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output Box")
    greet_btn = gr.Button("Greet")
    greet_btn.click(fn=greet, inputs=name, outputs=output)

demo.launch()