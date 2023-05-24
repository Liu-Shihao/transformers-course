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

其他pip命令
```shell
pip install Flask==1.0 
pip install Django>=1.11 
pip install numpy>=1.15,<=1.19
#更新依赖版本
pip install --upgrade requests
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
   
# Proxy 
## 代码中设置全局代理
通过设置 `os.environ` 的方式，代理配置将应用于整个 Python 进程，包括第三方库的网络请求部分。这样，当你运行 Huggingface 相关的代码时，它将自动使用代理进行网络连接。
```python
import os

# 设置代理配置
os.environ['http_proxy'] = 'http://your_proxy_address:proxy_port'
os.environ['https_proxy'] = 'http://your_proxy_address:proxy_port'
```
## corporate.pac
`corporate.pac` 文件是一个 PAC（Proxy Auto-Config）文件，它用于自动配置代理服务器的行为。PAC 文件是一个 JavaScript 文件，它包含一些规则和逻辑，用于确定特定 URL 的访问是否需要通过代理服务器。

当你的系统或浏览器配置为使用自动代理配置（Automatic Proxy Configuration）时，会使用 PAC 文件来判断是否要通过代理服务器访问特定的 URL。

PAC 文件中的规则可以基于 URL 的模式、主机名、端口等条件进行匹配和判断。根据规则的结果，代理服务器可以被动态地选择或绕过。

通过使用 PAC 文件，可以实现根据特定条件自动切换代理服务器，以满足不同的网络访问需求。这对于企业内部网络和外部网络的访问控制以及优化网络流量分配非常有用。

在你的情况下，你从 Windows 的代理配置中获取到的 corporate.pac 文件应该包含了你公司内部网络访问的代理规则。这个文件可能会指定某些 URL 需要通过代理访问，而其他 URL 可以直接访问或绕过代理。

请注意，PAC 文件是一个由网络管理员配置和管理的文件，对于具体的规则和逻辑，你可能需要联系你的网络管理员或查阅公司的相关文档来获取更多信息。
## 使用 PAC 文件设置代理的示例代码
使用 PAC 文件来设置代理的方式与直接指定代理地址和端口略有不同。你需要使用 urllib 库中的 urllib.request 模块来加载 PAC 文件并解析其规则，然后根据规则来确定要使用的代理配置。
```python
import urllib.request

def get_proxy(url):
    proxy_url = urllib.request.getproxies().get(url)
    if proxy_url:
        return urllib.request.ProxyHandler({'http': proxy_url, 'https': proxy_url})
    return urllib.request.ProxyHandler({})

# 加载 PAC 文件
pac_url = 'file:///path/to/corporate.pac'
proxy_handler = urllib.request.ProxyHandler({'http': pac_url, 'https': pac_url})

# 构建 opener
opener = urllib.request.build_opener(proxy_handler)

# 打开 URL
url = 'https://www.example.com'
response = opener.open(url)

# 使用代理进行网络请求
data = response.read()

# 处理响应数据
# ...

```
在设置代理的用户名和密码时，通常需要将其进行 Base64 编码。这是因为代理服务器在进行认证时，通常会要求提供经过编码的凭据。

在代码中设置代理时，你需要将用户名和密码进行 Base64 编码，并在代理设置中使用编码后的凭据。以下是一个示例：
```python
import urllib.request
import base64

# 设置代理服务器地址和端口
proxy_server = 'http://proxy_server:proxy_port'

# 设置代理服务器的认证信息
proxy_username = 'your_username'
proxy_password = 'your_password'

# 对用户名和密码进行 Base64 编码
credentials = base64.b64encode(f'{proxy_username}:{proxy_password}'.encode('utf-8')).decode('utf-8')

# 构建代理处理器
proxy_handler = urllib.request.ProxyHandler({'http': proxy_server, 'https': proxy_server})

# 设置代理的认证信息
proxy_handler.addheaders = [('Proxy-Authorization', f'Basic {credentials}')]

# 创建 opener
opener = urllib.request.build_opener(proxy_handler)
urllib.request.install_opener(opener)

# 现在可以通过 urllib 请求来使用代理
response = urllib.request.urlopen('http://example.com')

```
在上述示例中，proxy_username 和 proxy_password 分别是代理服务器的用户名和密码。我们使用 base64.b64encode() 方法将用户名和密码拼接成 username:password 的形式，并进行 Base64 编码。然后，在代理设置中，我们使用编码后的凭据添加了 Proxy-Authorization 头部，以进行认证。

请注意，使用 Base64 编码只是一种常见的方式，具体的认证方式可能因代理服务器和要使用的库而有所不同。在实际应用中，你可能需要根据代理服务器的要求和库的要求进行相应的编码和设置。