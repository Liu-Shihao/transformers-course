from transformers import BertConfig, BertModel
'''
https://huggingface.co/learn/nlp-course/zh-CN/chapter2/3?fw=pt
在本节中，我们将更详细地了解如何创建和使用模型。我们将使用 AutoModel类，当您希望从检查点实例化任何模型时，这非常方便。
这个AutoModel类及其所有相关项实际上是对库中各种可用模型的简单包装。它是一个聪明的包装器，因为它可以自动猜测检查点的适当模型体系结构，然后用该体系结构实例化模型。
但是，如果您知道要使用的模型类型，则可以使用直接定义其体系结构的类。让我们看看这是如何与BERT模型一起工作的。
'''

# Building the config
config = BertConfig()
# 初始化BERT模型需要做的第一件事是加载配置对象
# Building the model from the config
model = BertModel(config)
print(config)
'''
BertConfig {
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.28.1",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
'''