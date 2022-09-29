from dataclasses import replace
import json
from math import comb
import re
from threading import local
import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from itertools import combinations
import random
import time
import pickle
import os

def split_train_test(file_name):
    '''
    split json file into train, val, test
    '''
    with open(file_name, 'r') as origin:
        data = json.load(origin)

    doc_length = len(data)
    train_len = int(doc_length*0.8)
    left_length = doc_length-train_len

    if left_length % 2 == 0:
        val_len = left_length//2
    else:
        val_len = left_length//2

    random.shuffle(data)
    train_data = data[:train_len]
    val_data = data[train_len:train_len+val_len]
    test_data = data[train_len+val_len:]

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
    context_pos = []
    context_neg = []

    for doc_num, doc in enumerate(data):
        latexes = doc['latex']
        if len(latexes) < 2:
            continue

        for i in range(len(latexes)):
            # append query
            question.append(latexes[i])

            # append positive sample
            if i==(len(latexes)-1): # last item
                context_pos.append(latexes[0])
            else:
                context_pos.append(latexes[i+1])

            # append negative sample
            flag = True
            while flag:
                neg_index = np.random.randint(len(all_latex), size=num_neg)
                if doc_num not in neg_index:
                    neg_list = []
                    for ls in all_latex[neg_index]:
                        context_neg.append(random.choice(ls)) 
                    flag = False
    return question, context_pos, context_neg


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

def make_saved_dataset(file_name, num_neg, tokenizer, mode):
    start = time.time()
    question, context_pos, context_neg = make_negative_dataset(file_name, num_neg)
    neg = time.time()
    print(f'make neg {neg-start}')
    print(f'len q {len(question)} len pos {len(context_pos)} len neg {len(context_neg)}')

    q_seqs = tokenizer.encode_batch(question)
    encodeq = time.time()
    print(f'make q seqs {encodeq - neg}')    
    with open(f'data/{mode}/tokened_q.pkl', 'wb') as f:
        pickle.dump(q_seqs, f)
    del q_seqs

    p_pos_seqs = tokenizer.encode_batch(context_pos)
    encodepp = time.time()
    print(f'encode pp {encodepp-encodeq}')
    with open(f'data/{mode}/tokened_pp.pkl', 'wb') as f:
        pickle.dump(p_pos_seqs, f)
    del p_pos_seqs

    p_neg_seqs = tokenizer.encode_batch(context_neg)
    encodepn = time.time()
    print(f'encode pn {encodepn-encodepp}')
    with open(f'data/{mode}/tokened_pn.pkl', 'wb') as f:
        pickle.dump(p_neg_seqs, f)
    del p_neg_seqs

    with open(f'data/{mode}/tokened_q.pkl', 'rb') as f:
        q_seqs = pickle.load(f)
    start = time.time()
    q_dataset = make_tri_dataset_q(q_seqs) # dict of list
    del q_seqs
    triq = time.time()
    print(f'tri dataset q {triq-start}')
    with open(f'data/{mode}/tri_q.pkl', 'wb') as f:
        pickle.dump(q_dataset, f)
    del q_dataset

    with open(f'data/{mode}/tokened_pp.pkl', 'rb') as f:
        pp_seqs = pickle.load(f)
    pp_dataset = make_tri_dataset_q(pp_seqs) # dict of list
    del pp_seqs
    tripp = time.time()
    print(f'tri dataset pp {tripp-triq}')
    with open(f'data/{mode}/tri_pp.pkl', 'wb') as f:
        pickle.dump(pp_dataset, f)
    del pp_dataset

    with open(f'data/{mode}/tokened_pn.pkl', 'rb') as f:
        pn_seqs = pickle.load(f)
    pn_dataset = make_tri_dataset_q(pn_seqs) # dict of list
    del pn_seqs
    tripn = time.time()
    print(f'tri dataset pn {tripn-tripp}')
    with open(f'data/{mode}/tri_pn.pkl', 'wb') as f:
        pickle.dump(pn_dataset, f)
    del pn_dataset

# def make_saved_dataset_val(tokenizer, mode):
#     with open(f'data/{mode}_data.json', 'rb') as f:
#         data = json.load(f)
#     q_seqs = [tokenizer.encode_batch(doc['latex']) for doc in data]
#     q_dataset = make_tri_dataset_p(q_seqs) # dict of list
#     with open(f'data/{mode}/tri_dataset.pkl', 'wb') as f:
#         pickle.dump(q_dataset, f)

# def make_final_dataset_val(file_path):
#     with open(os.path.join(file_path, 'tri_dataset.pkl'), 'rb') as f:
#         q_dataset = pickle.load(f)
#     dataset = TensorDataset(q_dataset['input_ids'], q_dataset['token_type_ids'], q_dataset['attention_mask'])
#     return dataset

def make_final_dataset(file_path):
    with open(os.path.join(file_path, 'tri_q.pkl'), 'rb') as f:
        q_dataset = pickle.load(f)
    with open(os.path.join(file_path, 'tri_pp.pkl'), 'rb') as f:
        pp_dataset = pickle.load(f)
    with open(os.path.join(file_path, 'tri_pn.pkl'), 'rb') as f:
        pn_dataset = pickle.load(f)
    dataset = TensorDataset(q_dataset['input_ids'], q_dataset['token_type_ids'], q_dataset['attention_mask'], # (sample_size, 100)
                            pp_dataset['input_ids'], pp_dataset['token_type_ids'], pp_dataset['attention_mask'],
                            pn_dataset['input_ids'], pn_dataset['token_type_ids'], pn_dataset['attention_mask']) 
    return dataset

if __name__ == '__main__':
    import random
    np.random.seed(1004)
    random.seed(1004)
    from tokenizers import Tokenizer

    # # test split dataset
    # file_name = 'data/clean_anno.json'
    # split_train_test(file_name)

    tokenizer_path = 'data/tokenizer/tokenizer-bpe.json'
    tokenizer = Tokenizer.from_file(tokenizer_path)
    # for mode in ['train', 'val', 'test']:
    #     file_name = f'data/{mode}_data.json'
    #     dataset = make_saved_dataset(file_name, 1, tokenizer, mode)
    # file_path = 'data/train'
    # dataset = make_final_dataset(file_path)    
    # print(len(dataset[0]), dataset[0][0].shape, dataset[0][3].shape, dataset[0][6].shape)

    make_saved_dataset_val(tokenizer, 'val')