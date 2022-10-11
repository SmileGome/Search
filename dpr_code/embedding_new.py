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
    # doc_id = []

    for doc in dataset:
        for la in doc['latex']:
            latex.append(la)
            # doc_id.append(doc['id'])
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
    return val_dataset # doc_id

def save_docid(file_name):
    with open(file_name, 'r') as f:
        dataset = json.load(f)
    for doc in dataset:
        id = doc['id']
        for _ in range(len(doc['latex'])):
            with open("data/doc_id.txt", 'a') as f:
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
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            batch = tuple(t.to(device) for t in batch)

            inputs = {'input_ids': batch[0],
                    'token_type_ids': batch[1],
                    'attention_mask': batch[2]
                    }
            outputs = encoder(**inputs).pooler_output.tolist()
            with open('data/embeddings.txt', 'a') as f:
                f.write(str(outputs)+'\n')
            #   embs.append(outputs.detach())
        
            torch.cuda.empty_cache()
            # if i==2:
            #     break
    # embs = torch.concat(embs, dim=0).tolist()

    # for em, do in zip(embs, doc_id):
    #     with open('data/embeddings_76.txt', 'a') as f:
    #         f.write(str(do)+' '+str(em)+'\n')
    
def docid_emb(doc_file, emb_file):
    final = open('data/doc_emb.txt', 'a')
    d = open(doc_file, 'r')
    e = open(emb_file, 'r')
    d_line = d.readline()
    e_line = e.readline()
    while e_line!='':
        d_line = d.readline()
        e_line = d.readline()
        final.write(d_line+'\n'+e_line+'\n')
        


if __name__=="__main__":
#   embedding("data/clean_anno.json", "data/tokenizer/tokenizer-bpe.json", 'model/ex/ep4_acc0.769')
    
# tokenizer_path = "data/tokenizer/tokenizer-bpe.json"
# tokenizer = Tokenizer.from_file(tokenizer_path)
# dataset, doc_id = make_dataset("data/clean_anno.json", tokenizer)
    save_docid("data/clean_anno.json")