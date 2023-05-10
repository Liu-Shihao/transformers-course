import gradio as gr
'''
多输入和输出组件
假设您有一个更复杂的函数，有多个输入和输出。在下面的示例中，我们定义了一个函数，该函数接受字符串、布尔值和数字，并返回字符串和数字。观察应该如何传递输入和输出组件列表。
您只需将组件包装在列表中。输入列表inputs中的每个组件依次对应函数的一个参数。输出列表outputs中的每个组件都对应于函数的一个返回值，两者均按顺序对应。
'''
def greet(name, is_morning, temperature):
    salutation = "Good morning" if is_morning else "Good evening"
    greeting = f"{salutation} {name}. It is {temperature} degrees today"
    celsius = (temperature - 32) * 5 / 9
    return greeting, round(celsius, 2)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "checkbox", gr.Slider(0, 100)],
    outputs=["text", "number"],
)
demo.launch()