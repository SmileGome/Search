import json
from threading import local
import numpy as np
import torch
from torch.utils.data import TensorDataset

def split_train_test(file_name):
  with open(file_name, 'r') as origin:
    data = json.load(origin)
  
  doc_length = len(data['latex_anno'])
  train_len = int(doc_length*0.8)
  left_length = doc_length-train_len
  
  if left_length%2 == 0:
    val_len, test_len = left_length//2, left_length//2
  else: val_len, test_len = left_length//2, left_length-(left_length//2) 
  
  # all_doc = np.array(data['latex_anno'])
  # idx1, idx2, idx3 = np.random.randint(, size=train_len)
  train_data = data['latex_anno'][:train_len]
  val_data =  data['latex_anno'][train_len:train_len+val_len]
  test_data = data['latex_anno'][train_len+val_len:]

  for d in ['train_data', 'val_data', 'test_data']:
    with open('data/'+d+'.json', 'w') as result:
      json.dump(locals()[d], result)
  

def make_negative_dataset(file_name, num_neg):
  with open(file_name) as f:
      data = json.load(f)
  all_latex = np.array([' '.join(doc['latex']) for doc in data])
  question = []
  context = []

  for doc_num, doc in enumerate(data):
    latexes = doc['latex']
    if len(latexes)==1:
      continue
    leng = set(range(len(latexes)))
    for idx, l in enumerate(latexes):
      # append query
      question.append(l)

      # append positive sample
      index = np.array(list(leng-{idx}))
      context.append(' '.join(list(np.array(latexes)[index])))

      # append negative sample
      flag = True
      while flag:
        neg_index = np.random.randint(len(all_latex), size=num_neg)
        if doc_num not in neg_index:
          context.extend(all_latex[neg_index])
          flag = False
  return question, context  


# anno.json -> dataset format
def make_dataset(file_name):
  with open(file_name) as f:
    data = json.load(f)

  question = []
  context = []

  for doc in data:
    latexes = doc['latex']
    if len(latexes)==1:
      continue
    leng = set(range(len(latexes)))
    for idx, l in enumerate(latexes):
      question.append(l)
      index = np.array(list(leng-{idx}))
      context.append(' '.join(list(np.array(latexes)[index])))
  return question, context  


def make_tri_dataset(seqs):
  tri_dataset = {}

  for idx, encoding in enumerate(seqs):
    if idx==0:
      tri_dataset['input_ids'] = torch.tensor(encoding.ids).view(1,-1)
      tri_dataset['token_type_ids'] = torch.tensor(encoding.type_ids).view(1,-1)
      tri_dataset['attention_mask'] = torch.tensor(encoding.attention_mask).view(1,-1)
      
    else:
      tri_dataset['input_ids'] = torch.cat([tri_dataset['input_ids'], torch.tensor(encoding.ids).view(1,-1)], dim=0)
      tri_dataset['token_type_ids'] = torch.cat([tri_dataset['token_type_ids'], torch.tensor(encoding.type_ids).view(1,-1)], dim=0)
      tri_dataset['attention_mask'] = torch.cat([tri_dataset['attention_mask'], torch.tensor(encoding.attention_mask).view(1,-1)], dim=0)
  return tri_dataset

def make_tensor_dataset(p_dataset, q_dataset, num_neg):
  sample_size = q_dataset['input_ids'].shape[0]
  p_dataset['input_ids'] = p_dataset['input_ids'].view(sample_size, (num_neg+1), -1)
  p_dataset['attention_mask'] = p_dataset['attention_mask'].view(sample_size, (num_neg+1), -1)
  p_dataset['token_type_ids'] = p_dataset['token_type_ids'].view(sample_size, (num_neg+1), -1)

  dataset = TensorDataset(p_dataset['input_ids'], p_dataset['attention_mask'], p_dataset['token_type_ids'], 
                          q_dataset['input_ids'], q_dataset['attention_mask'], q_dataset['token_type_ids'])
  return dataset

def make_final_dataset(file_name, num_neg, tokenizer):
  question, context = make_negative_dataset(file_name, num_neg)
  q_seqs, p_seqs = tokenizer.encode_batch(question), tokenizer.encode_batch(context)
  q_dataset, p_dataset = make_tri_dataset(q_seqs), make_tri_dataset(p_seqs)
  dataset = make_tensor_dataset(p_dataset, q_dataset, num_neg)
  return dataset


if __name__=='__main__':
  file_name = 'data/anno.json'
  # num_neg = 2
  # question, context = make_negative_dataset(file_name, num_neg)
  # print(len(question))
  # print(len(context))
  split_train_test('data/anno.json')