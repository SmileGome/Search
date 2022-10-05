# set seed
import numpy as np
import random
from tokenizers import Tokenizer
import torch
from transformers import DPRConfig, DPRQuestionEncoder, TrainingArguments, get_linear_schedule_with_warmup, AdamW
from tqdm import tqdm
import wandb
import argparse
import yaml
import json
import pickle


def make_embedding(file_name, model_path, tokenizer_path="data/tokenizer/tokenizer-bpe.json"):
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


    with open(file_name, "rb") as f:
        dataset = json.load(f)
    for i, doc in enumerate(tqdm(dataset)):
        lst = []
        for la in tqdm(doc['latex']):
            tla = tokenizer.encode(la)
            with torch.no_grad():
                input = {}
                input['input_ids'] = torch.tensor(tla.ids).unsqueeze(dim=0).to(device)
                input['token_type_ids'] = torch.tensor(tla.type_ids).unsqueeze(dim=0).to(device)
                input['attention_mask'] = torch.tensor(tla.attention_mask).unsqueeze(dim=0).to(device)
                output = encoder(**input).pooler_output
                lst.append(output.detach())
        doc['embedding'] = lst

    with open("data/clean_anno_emb.json", "wb") as f:
        json.dump(dataset, f)


        # dataset = TensorDataset(
        # tri_dataset['input_ids'], tri_dataset['token_type_ids'], tri_dataset['attention_mask'])
        # dataloader = DataLoader(
        # dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        # with torch.no_grad():
        #     for batch in dataloader:
        #         batch = tuple(t.to(device) for t in batch)
        #         encoder



        # if len(tri_dataset['input_ids']) < batch_size:
            # for key in tri_dataset:
            #     tri_dataset[key] = tri_dataset[key].to(device)

if __name__=='__main__':
    make_embedding('data/clean_anno.json', 'model/ep4_acc0.769')