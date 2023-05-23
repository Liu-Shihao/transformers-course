# 什么是自然语言处理？
NLP 是语言学和机器学习交叉领域，专注于理解与人类语言相关的一切。 NLP 任务的目标不仅是单独理解单个单词，而且是能够理解这些单词的上下文。

以下是常见 NLP 任务的列表，每个任务都有一些示例：

- 对整个句子进行分类: 获取评论的情绪，检测电子邮件是否为垃圾邮件，确定句子在语法上是否正确或两个句子在逻辑上是否相关
- 对句子中的每个词进行分类: 识别句子的语法成分（名词、动词、形容词）或命名实体（人、地点、组织）
- 生成文本内容: 用自动生成的文本完成提示，用屏蔽词填充文本中的空白
- 从文本中提取答案: 给定问题和上下文，根据上下文中提供的信息提取问题的答案
- 从输入文本生成新句子: 将文本翻译成另一种语言，总结文本

NLP 不仅限于书面文本。它还解决了语音识别和计算机视觉中的复杂挑战，例如生成音频样本的转录或图像描述。

#pipeline
🤗 Transformers 库中最基本的对象是 pipeline() 函数。它将模型与其必要的预处理和后处理步骤连接起来，使我们能够通过直接输入任何文本并获得最终的答案：

# model

# tokenizer
与其他神经网络一样，Transformer模型无法直接处理原始文本， 因此我们管道的第一步是将文本输入转换为模型能够理解的数字。 为此，我们使用tokenizer(标记器)

# 链接🔗
- https://huggingface.co/models

深度学习入门课程，例如DeepLearning.AI 提供的 fast.ai实用深度学习教程
- https://www.deeplearning.ai/ 
- https://course.fast.ai/

PyTorch & TensorFlow 
- https://pytorch.org/
- https://www.tensorflow.org/

DeepLearning.AI的自然语言处理系列课程，其中涵盖了广泛的传统 NLP 模型，如朴素贝叶斯和 LSTM
- https://www.coursera.org/specializations/natural-language-processing?utm_source=deeplearning-ai&utm_medium=institutions&utm_campaign=20211011-nlp-2-hugging_face-page-nlp-refresh


安装依赖
```shell
使用 pip 包管理器安装 🤗 Transformers 的开发版本
pip install "transformers[sentencepiece]"
pip install -r requirements.txt
```

初始环境
```shell
mkdir ~/transformers-course
cd ~/transformers-course
#在这个目录中，使用 Python venv 模块创建一个虚拟环境：
python -m venv .env

#文件夹中看到一个名为 .env 的目录
ls -a

#可以使用activate和deactivate命令来控制进入和退出您的虚拟环境
# Activate the virtual environment
source venv/bin/activate

# Deactivate the virtual environment
source venv/bin/deactivate

#可以通过运行 which python 命令来检测虚拟环境是否被激活
which python


```

# 安装 TensorFlow 2.0 或 PyTorch 环境
TensorFlow和PyTorch都是当前非常流行的深度学习框架，它们都提供了一种方便的方式来构建、训练和部署深度神经网络模型。
TensorFlow是由Google开发的深度学习框架，它可以在多种平台上运行，包括CPU、GPU和TPU（Tensor Processing Units）。TensorFlow提供了一种数据流编程模型，可以用来构建各种各样的神经网络，例如卷积神经网络（CNNs）、递归神经网络（RNNs）和生成对抗网络（GANs）。TensorFlow的优点之一是具有非常丰富的社区支持，因此有很多高质量的资源、工具和库可用。

PyTorch是由Facebook开发的深度学习框架，它提供了一种动态图计算的方式来构建神经网络，这使得它在处理复杂的、可变的数据结构时更加方便。与TensorFlow不同，PyTorch是使用Python编写的，这使得它更容易上手，并且有更加Pythonic的API。PyTorch还提供了一个名为TorchScript的工具，可以将训练好的模型转换为高效的C++代码，以便在生产环境中进行部署。

无论是TensorFlow还是PyTorch都具有广泛的应用领域，例如计算机视觉、自然语言处理和语音识别。选择哪个框架取决于您的需求和偏好，但是这两个框架都是学习深度学习的绝佳选择。

RuntimeError: At least one of TensorFlow 2.0 or PyTorch should be installed. To install TensorFlow 2.0, read the instructions at https://www.tensorflow.org/install/ To install PyTorch, read the instructions at https://pytorch.org/.

https://pytorch.org/get-started/locally/#macos-version

https://www.tensorflow.org/install?hl=zh-cn

```shell
#使用pip安装
pip3 install torch torchvision

#使用Macos Command安装
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
sh Miniconda3-latest-MacOSX-arm64.sh

curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
sh Miniconda3-latest-MacOSX-x86_64.sh
```



# Error
1. ValueError: This tokenizer cannot be instantiated. Please make sure you have `sentencepiece` installed in order to use this tokenizer.
这个错误表明你需要安装 sentencepiece 库才能使用这个tokenizer。你可以在终端中使用以下命令来安装：pip install sentencepiece

2. RuntimeError: At least one of TensorFlow 2.0 or PyTorch should be installed. To install TensorFlow 2.0, read the instructions at https://www.tensorflow.org/install/ To install PyTorch, read the instructions at https://pytorch.org/.
这个错误表明你需要安装TensorFlow或者PyTorch环境， 你可以在终端中使用以下命令来安装：pip3 install torch torchvision
   
3. ImportError: 
T5Converter requires the protobuf library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
 你可以在终端中使用以下命令来安装：pip install protobuf

4. TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
  Downgrade the protobuf package to 3.20.x or lower.
  Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
这个错误通常是由于在使用较新版本的 protobuf 库时，与生成的 protobuf 文件不兼容导致的。可以尝试将 protobuf 包降级到 3.20.x 或更低版本。您可以在终端中输入 pip install protobuf==3.20 来安装 3.20.x 版本的 protobuf。
你可以在终端中使用以下命令来安装：pip install protobuf==3.20
   
# LangChain

文档：https://python.langchain.com/en/latest/

Github：https://github.com/hwchase17/langchain
# Chroma

Chroma 是一个开源嵌入式数据库，宣称是使用内存构建 Python 或 JavaScript LLM 应用程序的最快方法。

安装简单、功能丰富、集成功能丰富（Langchain、LlamaIndex、OpenAI）、开源免费、有JS客户端

官网：https://www.trychroma.com/

GIthub：https://github.com/chroma-core/chroma

文档：https://docs.trychroma.com/
# Gradio

Create UIs for your machine learning model in Python in 3 minutes

Github地址：https://github.com/gradio-app/gradio

文档：https://www.gradio.app/quickstart/