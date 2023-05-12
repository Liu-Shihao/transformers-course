from langchain.document_loaders.bilibili import BiliBiliLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import GitLoader, TextLoader, PythonLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import JSONLoader
from git import Repo
import json
from pathlib import Path
from pprint import pprint
"""
https://github.com/Unstructured-IO/unstructured
Unstructured : python package. This package is a great way to transform all types of files - text, powerpoint, images, html, pdf, etc - into text data.

https://python.langchain.com/en/latest/modules/indexes/document_loaders.html
例如：txt、json、markdown、pdf、csv、bilibili、youtube、twitter、git、huggingface dateset 、url、file directory

!pip install bilibili-api
!pip install GitPython
!pip install pypdf
!pip install jq
"""

# 加载B站资源
loader = BiliBiliLoader(
    ["https://www.bilibili.com/video/BV1xt411o7Xu/"]
)
loader.load()

# 加载CSV文件
loader = CSVLoader(file_path='./example_data/mlb_teams_2012.csv')
data = loader.load()



# 加载Github
repo = Repo.clone_from(
    "https://github.com/hwchase17/langchain", to_path="./example_data/test_repo1"
)
# 从本地磁盘加载Github仓库
branch = repo.head.reference
loader = GitLoader(repo_path="./example_data/test_repo1/", branch=branch)
data = loader.load()
# 从url加载github仓库
loader = GitLoader(
    clone_url="https://github.com/hwchase17/langchain",
    repo_path="./example_data/test_repo2/",
    branch="master",
)
# 过滤文件
# eg. loading only python files
loader = GitLoader(repo_path="./example_data/test_repo1/", file_filter=lambda file_path: file_path.endswith(".py"))

# 加载PDF文件
loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
pages = loader.load_and_split()

# 加载文件夹
loader = DirectoryLoader('../', glob="**/*.md")
# Change loader class
loader = DirectoryLoader('../', glob="**/*.md", loader_cls=TextLoader)
loader = DirectoryLoader('../../../../../', glob="**/*.py", loader_cls=PythonLoader)
docs = loader.load()

# 加载json文件
file_path='./example_data/facebook_chat.json'
data = json.loads(Path(file_path).read_text())

loader = JSONLoader(
    file_path='../../example_data/facebook_chat.json',
    jq_schema='.messages[].content')

data = loader.load()

# 加载html文文件
loader = UnstructuredHTMLLoader("example_data/fake-content.html")
data = loader.load()