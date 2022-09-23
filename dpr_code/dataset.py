import json
from math import comb
import re
from threading import local
import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from itertools import combinations
import random


def split_train_test(file_name):
    '''
    split json file into train, val, test
    '''
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
    '''
    return 1-dim list(query), 2-dim list(context)
    '''
    with open(file_name) as f:
        data = json.load(f)
    all_latex = np.array([doc['latex'] for doc in data], dtype='object') # array of list
    question = []
    context = []

    for doc_num, doc in enumerate(data):
        latexes = doc['latex']
        if len(latexes) < 2:
            continue

        # make combination
        items = range(len(latexes))
        comb = list(combinations(items, 2))
        
        for (i, ii) in comb:
            # append query
            question.append(latexes[i])

            ith_lst = []
            # append positive sample
            ith_lst.append(latexes[ii])

            # append negative sample
            flag = True
            while flag:
                neg_index = np.random.randint(len(all_latex), size=num_neg)
                if doc_num not in neg_index:
                    neg_list = []
                    for ls in all_latex[neg_index]:
                        ith_lst.append(random.choice(ls)) 
                    flag = False
            context.append(ith_lst)
    return question, context


def make_tri_dataset_q(seqs):
    '''
    return dict of 1-dim tensor
    '''
    a = []
    b = []
    c = []

    for v in seqs:
        a.append(v.ids)
        b.append(v.type_ids)
        c.append(v.attention_mask)
    tri_dataset = {'input_ids': torch.tensor(a), 'token_type_ids': torch.tensor(b), 'attention_mask': torch.tensor(c)}
    return tri_dataset


def make_tri_dataset_p(seqs):
    '''
    return dict of 2-dim tensor
    '''
    a = seqs
    b = seqs
    c = seqs
    for i, v in enumerate(seqs):
        for ii, vv in enumerate(v):
            a[i][ii] = vv.ids
            b[i][ii] = vv.type_ids
            c[i][ii] = vv.attention_mask
    tri_dataset = {'input_ids': torch.tensor(a), 'token_type_ids': torch.tensor(b), 'attention_mask': torch.tensor(c)}
    return tri_dataset

def make_final_dataset(file_name, num_neg, tokenizer):
    question, context = make_negative_dataset(file_name, num_neg)
    q_seqs = tokenizer.encode_batch(question)    
    p_seqs = [tokenizer.encode_batch(c) for c in context]
    q_dataset = make_tri_dataset_q(q_seqs) # dict of list
    p_dataset = make_tri_dataset_p(p_seqs) # dict of list
    dataset = TensorDataset(q_dataset['input_ids'], q_dataset['token_type_ids'], q_dataset['attention_mask'], # (sample_size, 100)
                            p_dataset['input_ids'], p_dataset['token_type_ids'], p_dataset['attention_mask']) # (sample_size, 3, 100)
    return dataset


if __name__ == '__main__':
    import random
    from tokenizers import Tokenizer
    # np.random.seed(1004)
    # random.seed(1004)

    file_name = 'data/train_data.json'
    tokenizer_path = 'data/tokenizer-wordlevel.json'
    tokenizer = Tokenizer.from_file(tokenizer_path)

    dataset = make_final_dataset(file_name, 2, tokenizer)
    print(len(dataset[0]), dataset[0][0].shape, dataset[0][3].shape)