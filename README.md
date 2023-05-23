# Set Up
```shell
#创建一个虚拟环境：
python -m venv .env

#文件夹中看到一个名为 .env 的目录
ls -a

#可以使用activate和deactivate命令来控制进入和退出虚拟环境
# 在 macOS 和 Linux 上，可以运行以下命令：
source venv/bin/activate
#在 Windows 上，可以运行以下命令：
source myenv\Scripts\activate


# Deactivate the virtual environment
source venv/bin/deactivate

#可以通过运行 which python 命令来检测虚拟环境是否被激活
which python

#安装必需依赖
pip install -r requirements.txt
```

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
   
