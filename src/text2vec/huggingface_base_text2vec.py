from transformers import BertTokenizer, BertModel
import torch

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
model_name = "/Users/liushihao/PycharmProjects/model-hub/shibing624/text2vec-base-chinese"

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']
# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
# Perform pooling. In this case, mean pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
print("Sentence embeddings:")
print(sentence_embeddings)
'''
Sentence embeddings:
tensor([[-4.4425e-04, -2.9735e-01,  8.5790e-01,  ..., -5.2770e-01,
         -1.4316e-01, -1.0008e-01],
        [ 6.5362e-01, -7.6667e-02,  9.5962e-01,  ..., -6.0122e-01,
         -1.6790e-03,  2.1458e-01]])
'''
