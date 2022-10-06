# from transformers import DPRConfig, DPRQuestionEncoder, AutoModel, AutoConfig
# from torchsummary import summary
# # 'facebook/dpr-question_encoder-single-nq-base'
# model_path = "model/ep4_acc0.769"
# # model_path = 'facebook/dpr-question_encoder-single-nq-base'

# # load model
# config = DPRConfig.from_pretrained(model_path)
# # encoder = DPRQuestionEncoder.from_pretrained(
# #     model_path, config=config, 
# #     # ignore_mismatched_sizes=True
# #     )
# print(config)
# print('-'*50)
# # print(encoder)
# # summary(encoder,input_size=(768,),depth=1,batch_dim=1, dtypes=['torch.IntTensor']) 


# from torch.utils.data import DataLoader, 

import json

with open('data/clean_anno.json', 'r') as f:
  dataset = json.load(f)

# max_len = 0
# for doc in dataset:
#   length = len(doc['latex'])
#   if length > max_len:
#     max_len = length

# print(max_len)
a = dataset[0]['embedding']
b = dataset[0]['latex']
print(type(a), len(a), len(b), len(a[0]))