# set seed
from tokenizers import Tokenizer
import torch
from transformers import DPRConfig, DPRQuestionEncoder
from tqdm import tqdm
import json
from dataset import make_tri_dataset_q
from torch.utils.data import DataLoader, TensorDataset
import time

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

    # idx 단위로 접근 doc단위의 latex 접근,
    # dataset 삭제,
    # 다시 json 불러서(skiprow?) 그 부분만 수정
    s = time.time()
    with open(file_name, "r") as f:
        dataset = json.load(f)
    all_len = len(dataset)
    del dataset
    batch_size = 8
    for idx in tqdm(range(all_len)):
        result = []
        with open(file_name, "r") as f:
            dataset = json.load(f)
        # doc = dataset[idx]
        latexes = dataset[idx]['latex']
        # del dataset
        t_la = tokenizer.encode_batch(latexes)
        t_la = make_tri_dataset_q(t_la)
        tensordataset = TensorDataset(
            t_la['input_ids'], t_la['token_type_ids'], t_la['attention_mask'])
        dataloader = DataLoader(
            tensordataset, batch_size=batch_size, shuffle=False, drop_last=False)
        with torch.no_grad():
            for batch in dataloader:
                batch = tuple(t.to(device) for t in batch)
                input = {'input_ids': batch[0],
                         'token_type_ids': batch[1],
                         'attention_mask': batch[2]
                         }
                output = encoder(**input).pooler_output
                result.append(output.detach())
        result = torch.concat(result, dim=0).tolist()
        dataset[idx]['embedding'] = result
        del batch
        del latexes
        del output
        del result
        # with open(file_name, "r") as f:
        #     dataset = json.load(f)
        # dataset[idx] = doc
        # break
        with open(file_name, "w") as f:
            json.dump(dataset, f)


    # for i, doc in enumerate(tqdm(dataset)):
    #     lst = []
    #     for la in tqdm(doc['latex']):
    #         tla = tokenizer.encode(la)
    #         with torch.no_grad():
    #             input = {}
    #             input['input_ids'] = torch.tensor(tla.ids).unsqueeze(dim=0).to(device)
    #             input['token_type_ids'] = torch.tensor(tla.type_ids).unsqueeze(dim=0).to(device)
    #             input['attention_mask'] = torch.tensor(tla.attention_mask).unsqueeze(dim=0).to(device)
    #             output = encoder(**input).pooler_output
    #             lst.append(output.detach())
    #     doc['embedding'] = lst

    # with open("data/clean_anno_emb.json", "wb") as f:
    #     json.dump(dataset, f)

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


if __name__ == '__main__':
    make_embedding('data/clean_anno.json', 'model/ex/ep4_acc0.769')
