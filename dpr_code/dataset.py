import json
import re
from threading import local
import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset


def split_train_test(file_name):
    with open(file_name, 'r') as origin:
        data = json.load(origin)

    doc_length = len(data['latex_anno'])
    train_len = int(doc_length*0.8)
    left_length = doc_length-train_len

    if left_length % 2 == 0:
        val_len, test_len = left_length//2, left_length//2
    else:
        val_len, test_len = left_length//2, left_length-(left_length//2)

    # all_doc = np.array(data['latex_anno'])
    # idx1, idx2, idx3 = np.random.randint(, size=train_len)
    train_data = data['latex_anno'][:train_len]
    val_data = data['latex_anno'][train_len:train_len+val_len]
    test_data = data['latex_anno'][train_len+val_len:]

    for d in ['train_data', 'val_data', 'test_data']:
        with open('data/'+d+'.json', 'w') as result:
            json.dump(locals()[d], result)


def make_negative_dataset(file_name, num_neg):
    with open(file_name) as f:
        data = json.load(f)
    all_latex = np.array([doc['latex'] for doc in data], dtype='object')
    question = []
    context = []

    for doc_num, doc in enumerate(data):
        latexes = doc['latex']
        if len(latexes) == 1:
            continue
        leng = set(range(len(latexes)))
        for idx, l in enumerate(latexes):
            # append query
            question.append(l)

            # append positive sample
            index = np.array(list(leng-{idx}))
            # list로 엮인 positive latexes
            ith_lst = []
            ith_lst.append(list(np.array(latexes)[index]))

            # append negative sample
            flag = True
            while flag:
                neg_index = np.random.randint(len(all_latex), size=num_neg)
                if doc_num not in neg_index:
                    # list로 엮인 negative latexes
                    all_list = []
                    for l in all_latex[neg_index]:
                        all_list += l
                    ith_lst.append(all_list)
                    flag = False
            context.append(ith_lst)
    return question, context

def make_tri_dataset_q(seqs):
    a = []
    b = []
    c = []

    for v in seqs:
        a.append(v.ids)
        b.append(v.type_ids)
        c.append(v.attention_mask)
    tri_dataset = {'input_ids': a, 'token_type_ids': b, 'attention_mask': c}
    return tri_dataset


def make_tri_dataset_p(seqs):
    a = seqs
    b = seqs
    c = seqs
    for i, v in enumerate(seqs):
        for ii, vv in enumerate(v):
            for iii, vvv in enumerate(vv):
                a[i][ii][iii] = getattr(vvv, 'ids')
                b[i][ii][iii] = getattr(vvv, 'type_ids')
                c[i][ii][iii] = getattr(vvv, 'attention_mask')
    tri_dataset = {'input_ids': a, 'token_type_ids': b, 'attention_mask': c}
    return tri_dataset

class LatexDataset(Dataset):
    def __init__(self, tri_q, tri_p):
        self.tri_q = tri_q
        self.tri_p = tri_p
    
    def __getitem__(self, idx):
        query = {}
        context = {}  
        for key in self.tri_q.keys():
            q = torch.tensor(self.tri_q[key][idx])
            query[key] = q
            pos_context = torch.tensor(self.tri_p[key][idx][0])
            neg_context = torch.tensor(self.tri_p[key][idx][1])
            context[key] = [pos_context, neg_context]
        return query, context 

    def __len__(self):
        return len(self.tri_q) # query 개수

def make_final_dataset(file_name, num_neg, tokenizer):
    question, context = make_negative_dataset(file_name, num_neg)
    q_seqs = tokenizer.encode_batch(question)
    p_seqs = []
    for pn_set in context:
        ith_lst = []
        for lst in pn_set:
            ith_lst.append(tokenizer.encode_batch(lst))
        p_seqs.append(ith_lst)

    q_dataset = make_tri_dataset_q(q_seqs) # dict of list
    p_dataset = make_tri_dataset_p(p_seqs) # dict of list
    
    return q_dataset, p_dataset


if __name__ == '__main__':
    import random
    from tokenizers import Tokenizer
    np.random.seed(1004)
    random.seed(1004)

    file_name = 'data/train_data.json'
    tokenizer_path = 'data/tokenizer-wordlevel.json'
    tokenizer = Tokenizer.from_file(tokenizer_path)
    q_dataset, p_dataset = make_final_dataset(file_name, 2, tokenizer)
    dataset = LatexDataset(tri_q=q_dataset, tri_p=p_dataset)
    q, p = dataset[0] # q : dict of tensor, p : dict of list
    for k, v in q.items():
        print(f'{k}: {v}')

    for k, v in p.items():
        print(f'{k}: {len(v)}')
        for i in v:
            print(i.shape)