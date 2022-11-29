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

# import json

# with open('data/embeddings.txt', 'r') as f:
#   lines = f.readlines()
#   print(len(lines))

# with open('data/doc_id.txt', 'r') as f:
#   lines = f.readlines()
#   print(len(lines))

# with open('data/result/emb_short/doc_emb.txt', 'r') as f:
#   lines = f.readlines()
#   for i in range(4):
#     print(len(lines[i]))
#   print(len(lines))

# import json
# import pickle

# with open('data\clean_anno.json', 'r') as f:
#   data = json.load(f)
#   sample_latex = data[0]['latex'][0]
#   print(f'input: {sample_latex}')

import pickle
with open('data/result/backend_sample/backend_sample_input.pkl', 'rb') as f:
  lst = pickle.load(f)
print(len(lst), type(lst))