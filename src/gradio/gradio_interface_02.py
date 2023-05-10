import gradio as gr
'''
组件属性
在之前的示例中我们可以看到一些简单的文本框组件 Textbox ，但是如果您想改变UI组件的外观或行为呢?

假设您想要自定义输入文本字段，例如您希望它更大并有一个文本占位符。如果我们使用 Textbox 的实际类，而不是使用字符串快捷方式，就可以通过组件属性实现个性化。
'''

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(lines=2, placeholder="Name Here..."),
    outputs="text",
)
demo.launch()