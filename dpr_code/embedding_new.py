from dataset import make_tri_dataset_q
from tokenizers import Tokenizer
import torch
from transformers import DPRConfig, DPRQuestionEncoder
from tqdm import tqdm
import json
from dataset import make_tri_dataset_q
from torch.utils.data import DataLoader, TensorDataset


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

    tokenized_latex = tokenizer.encode_batch(latex)
    tri_dataset = make_tri_dataset_q(tokenized_latex)
    val_dataset = TensorDataset(
        tri_dataset['input_ids'], tri_dataset['token_type_ids'], tri_dataset['attention_mask'])
    return val_dataset, doc_id


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
    dataset, doc_id = make_dataset(file_name, tokenizer)
    dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        
    embs = []
    encoder.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
          batch = tuple(t.to(device) for t in batch)

          inputs = {'input_ids': batch[0],
                  'token_type_ids': batch[1],
                  'attention_mask': batch[2]
                  }
          outputs = encoder(**inputs).pooler_output
          embs.append(outputs.detach())
          torch.cuda.empty_cache()
          if i==2:
            break
    embs = torch.concat(embs, dim=0).tolist()

    for em, do in zip(embs, doc_id):
      with open('data/embeddings_76.txt', 'a') as f:
        f.write(str(do)+' '+str(em)+'\n')
    
if __name__=="__main__":
  embedding("data/clean_anno.json", "data/tokenizer/tokenizer-bpe.json", 'model/ex/ep4_acc0.769')