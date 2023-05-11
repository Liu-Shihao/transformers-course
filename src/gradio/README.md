#Gradio: 用Python构建机器学习网页APP
Create UIs for your machine learning model in Python in 3 minutes

Gradio是一个开源的Python库，用于构建演示机器学习或数据科学，以及web应用程序。

使用Gradio，您可以基于您的机器学习模型或数据科学工作流快速创建一个漂亮的用户界面，让用户可以”尝试“拖放他们自己的图像、粘贴文本、录制他们自己的声音，并通过浏览器与您的演示程序进行交互。

https://www.gradio.app/quickstart/
```shell
pip install gradio
```


```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
    
demo.launch()
```
