from dataset import make_tri_dataset_q
from tokenizers import Tokenizer
import torch
from transformers import DPRConfig, DPRQuestionEncoder
from tqdm import tqdm
import json
from dataset import make_tri_dataset_q
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys

'''
this file for making doc_emb.txt 
to save embeddings in elastic DB
'''


def make_dataset(file_name, tokenizer):
    # 파일불러서 tri dataset 만들기
    with open(file_name, 'r') as f:
        dataset = json.load(f)

    latex = []
    # doc_id = []

    for doc in dataset:
        for la in doc['latex']:
            latex.append(la)
            # doc_id.append(doc['id'])

    tokenized_latex = tokenizer.encode_batch(latex)

    tri_dataset = make_tri_dataset_q(tokenized_latex)
    dataset = TensorDataset(
        tri_dataset['input_ids'], tri_dataset['token_type_ids'], tri_dataset['attention_mask'])
    return dataset


def save_docid(file_name, output_path):
    '''
    doc_id 만들기
    '''
    with open(file_name, 'r') as f:
        dataset = json.load(f)
    with open(output_path, 'a') as f:
        for doc in dataset:
            id = doc['id']
            for _ in range(len(doc['latex'])):
                f.write(id+'\n')


def embedding(file_name, tokenizer_path, model_path, output_path):
    '''
    임베딩 만들기
    '''
    tokenizer = Tokenizer.from_file(tokenizer_path)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")

    # load model
    config = DPRConfig.from_pretrained(model_path)
    encoder = DPRQuestionEncoder.from_pretrained(
        model_path, config=config,
        ignore_mismatched_sizes=True
    )
    encoder = encoder.to(device)
    encoder.eval()

    batch_size = 8
    dataset = make_dataset(file_name, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    del dataset
    encoder.eval()
    cnt = 0
    f = open(output_path, 'a')
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            batch = tuple(t.to(device) for t in batch)

            inputs = {'input_ids': batch[0],
                      'token_type_ids': batch[1],
                      'attention_mask': batch[2]
                      }
            outputs = encoder(**inputs).pooler_output.tolist()
            cnt += len(batch[0])
            for o in outputs:
                f.write(str(o)+'\n')
            torch.cuda.empty_cache()
    print(f"all length : {cnt}")


def docid_emb(doc_file, emb_file, output_path):
    '''
    만들어놓은 doc id, emb 하나의 파일로 만들기
    '''
    final = open(output_path, 'w')
    d = open(doc_file, 'r')
    e = open(emb_file, 'r')
    e_line = '  '
    cnt = 0
    while e_line != '':
        d_line = d.readline()
        e_line = e.readline()
        final.write(d_line)
        final.write(e_line)
        cnt += 1
    print('done!!!', cnt)


if __name__ == "__main__":
    # step 1
    # save_docid("data/clean_anno.json", "data/doc_id.txt")
    
    # step 2
    # embedding("data/clean_anno.json", "data/tokenizer/tokenizer-bpe.json", 'model\small_pool_384_shortval\ep3_acc0.847', 'data/result/emb_short/embeddings.txt')

    # step 3
    docid_emb('data/result/doc_id.txt', 'data/result/emb_short/embeddings.txt', 'data/result/emb_short/doc_emb.txt')

    # tokenizer_path = "data/tokenizer/tokenizer-bpe.json"
    # tokenizer = Tokenizer.from_file(tokenizer_path)
    # dataset, doc_id = make_dataset("data/clean_anno.json", tokenizer)
    # print(len(dataset), len(doc_id))
