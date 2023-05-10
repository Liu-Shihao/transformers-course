from transformers import BertModel
'''
加载已经训练过的Transformers模型很简单-我们可以使用from_pretrained() 方法:model = BertModel.from_pretrained("bert-base-cased")
权重已下载并缓存在缓存文件夹中（因此将来对from_pretrained()方法的调用将不会重新下载它们）默认为 ~/.cache/huggingface/transformers . 您可以通过设置 HF_HOME 环境变量来自定义缓存文件夹。

保存模型和加载模型一样简单—我们使用 save_pretrained() 方法，类似于 from_pretrained() 方法：model.save_pretrained("directory_on_my_computer")
这会将两个文件保存到磁盘：config.json /pytorch_model.bin
如果你看一下 config.json 文件，您将识别构建模型体系结构所需的属性。该文件还包含一些元数据，例如检查点的来源以及上次保存检查点时使用的🤗 Transformers版本。

这个 pytorch_model.bin 文件就是众所周知的state dictionary; 它包含模型的所有权重。这两个文件齐头并进；配置是了解模型体系结构所必需的，而模型权重是模型的参数。
'''
model_name = "/Users/liushihao/PycharmProjects/model-hub/bert-base-cased"
model = BertModel.from_pretrained(model_name)