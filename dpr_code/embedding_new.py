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

def make_dataset(file_name, tokenizer):
    # 파일불러서 tri dataset 만들기
    with open(file_name, 'r') as f:
        dataset = json.load(f)

    latex = []
    doc_id = []

    for doc in dataset:
        for la in doc['latex']:
            latex.append(la)
            doc_id.append(doc['id'])
    # torch.tensor(np.array(doc_id))
    # print(f"latex : {sys.getsizeof(latex)}")
    # print(f"doc_id : {sys.getsizeof(doc_id)}")

    tokenized_latex = tokenizer.encode_batch(latex)
    # print(f"after tokenize : {sys.getsizeof(tokenized_latex)}")

    tri_dataset = make_tri_dataset_q(tokenized_latex)
    # print(f"tri_dataset: {sys.getsizeof(tri_dataset)}")
    val_dataset = TensorDataset(
        tri_dataset['input_ids'], tri_dataset['token_type_ids'], tri_dataset['attention_mask'])
    # print(f"final dataset : {sys.getsizeof(val_dataset)}")
    return val_dataset, doc_id

def save_docid(file_name):
    with open(file_name, 'r') as f:
        dataset = json.load(f)
    with open("data/doc_id.txt", 'a') as f:
        for doc in dataset:
            id = doc['id']
            for _ in range(len(doc['latex'])):
                    f.write(id+'\n')

def embedding(file_name, tokenizer_path, model_path):
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
    dataset, _ = make_dataset(file_name, tokenizer)
    dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    del dataset
    embs = []
    encoder.eval()
    cnt = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            batch = tuple(t.to(device) for t in batch)

            inputs = {'input_ids': batch[0],
                    'token_type_ids': batch[1],
                    'attention_mask': batch[2]
                    }
            outputs = encoder(**inputs).pooler_output.tolist()
            cnt += len(batch[0])
            with open('data/embeddings.txt', 'a') as f:
                for o in outputs:
                    f.write(str(o)+'\n')
            #   embs.append(outputs.detach())
        
            torch.cuda.empty_cache()
            # if i==2:
            #     break
    # embs = torch.concat(embs, dim=0).tolist()
    print(f"all length : {cnt}")

    # for em, do in zip(embs, doc_id):
    #     with open('data/embeddings_76.txt', 'a') as f:
    #         f.write(str(do)+' '+str(em)+'\n')
    
def docid_emb(doc_file, emb_file):
    final = open('data/doc_emb.txt', 'w')
    d = open(doc_file, 'r')
    e = open(emb_file, 'r')
    e_line = '  '
    cnt = 0
    while e_line!='':
        cnt += 1
        d_line = d.readline()
        e_line = e.readline()
        final.write(d_line)
        final.write(e_line)
    print('done!!!', cnt)
        


if __name__=="__main__":
    # save_docid("data/clean_anno.json")
    # embedding("data/clean_anno.json", "data/tokenizer/tokenizer-bpe.json", 'model/ex/ep4_acc0.769')
    
    docid_emb('data/doc_id.txt', 'data/embeddings.txt')

    # tokenizer_path = "data/tokenizer/tokenizer-bpe.json"
    # tokenizer = Tokenizer.from_file(tokenizer_path)
    # dataset, doc_id = make_dataset("data/clean_anno.json", tokenizer)
    # print(len(dataset), len(doc_id))
