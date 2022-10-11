# set seed
from dataset import make_final_dataset, make_tri_dataset_q

import numpy as np
import random
import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from transformers import DPRConfig, DPRQuestionEncoder, TrainingArguments, get_linear_schedule_with_warmup, AdamW
from tqdm import tqdm
import wandb
import argparse
import yaml
import json
import pickle


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def make_val_dataset(args, tokenizer):
    # 파일불러서 tri dataset 만들기
    with open(args['val_file_name'], 'r') as f:
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


def validation(encoder, device, val_dataloader, doc_id):
    embs = []
    encoder.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader)):
            # if i==(len(val_dataloader)//3): #########################modify
            #     break
            batch = tuple(t.to(device) for t in batch)

            inputs = {'input_ids': batch[0],
                    'token_type_ids': batch[1],
                    'attention_mask': batch[2]
                    }
            outputs = encoder(**inputs).pooler_output
            embs.append(outputs.detach())
            torch.cuda.empty_cache()

    embs = torch.concat(embs, dim=0)
    cos = CosineSimilarity(dim=1, eps=1e-6)
    cnt = 0
    for i, val in enumerate(tqdm(embs)):
        val = torch.unsqueeze(val, 0)
        sim_score = cos(val, embs).tolist()
        sim_score = [[s, i] for s, i in zip(sim_score, doc_id)]
        sim_score.sort(key=lambda x: x[0], reverse=True)
        top_doc = [i[1] for i in sim_score[1:11]]
        
        if doc_id[i] in top_doc:
            cnt += 1

    return cnt/len(embs)


def train(args):
    # seed
    seed_everything(args['seed'])

    # settings
    print("pytorch version: {}".format(torch.__version__))
    gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    print("GPU 사용 가능 여부: {}".format(gpu))
    print(torch.cuda.get_device_name(0))
    print(f'GPU 개수: {torch.cuda.device_count()}')
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    # load tokenizer
    tokenizer = Tokenizer.from_file(args['tokenizer_path'])

    # load dataset
    # split_train_test(args['all_file_name'])
    train_dataset = make_final_dataset(args['train_file_name'])
    val_dataset, doc_id = make_val_dataset(args, tokenizer)

    # make dataloader
    train_dataloader = DataLoader(
        train_dataset, batch_size=args['batch_size'], shuffle=True, drop_last=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args['batch_size'], shuffle=False, drop_last=False)

    # load model
    config = DPRConfig.from_pretrained(args['model_path'])
    encoder = DPRQuestionEncoder.from_pretrained(
        args['model_path'], config=config, 
        ignore_mismatched_sizes=True
        )
    encoder = encoder.to(device)

    encoder.config.decoder_start_token_id = tokenizer.token_to_id("[CLS]")
    encoder.config.pad_token_id = tokenizer.token_to_id("[PAD]")
    encoder.config.vocab_size = args['vocab_size']

    # set beam search parameters
    encoder.config.eos_token_id = tokenizer.token_to_id("[SEP]")
    encoder.config.max_length = args['max_length_token']
    encoder.config.early_stopping = True
    encoder.config.no_repeat_ngram_size = 3
    encoder.config.length_penalty = 2.0
    encoder.config.num_beams = 4
    encoder.config.hidden_size = args['hidden_size']

    # start training
    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in encoder.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
        {'params': [p for n, p in encoder.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args['learning_rate'], eps=args['adam_epsilon'])
    t_total = len(
        train_dataloader) // args['gradient_accumulation_steps'] * args['num_epoch']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps=t_total)

    step = 0
    best_acc = -1
    # loop over the dataset multiple times
    for epoch in range(args['num_epoch']):
        # train
        encoder.train()

        train_loss = 0.0
        for i, batch in enumerate(tqdm(train_dataloader)):
            batch = tuple(t.to(device) for t in batch)

            q_inputs = {'input_ids': batch[0],
                        'token_type_ids': batch[1],
                        'attention_mask': batch[2]
                        }

            p_inputs = {'input_ids': batch[3].view(args['batch_size']*(args['num_neg']+1), -1),
                        'token_type_ids': batch[4].view(args['batch_size']*(args['num_neg']+1), -1),
                        'attention_mask': batch[5].view(args['batch_size']*(args['num_neg']+1), -1)
                        }

            # (batch_size*, emb_dim)
            q_outputs = encoder(**q_inputs).pooler_output
            # (batch_size*(args['num_neg']+1), emb_dim)
            p_outputs = encoder(**p_inputs).pooler_output
            # (batch_size, 1, emb_dim)
            q_outputs = torch.unsqueeze(q_outputs, 1)
            # (batch_size, emb_dim, num_neg+1)
            p_outputs = p_outputs.reshape(
                args['batch_size'], -1, args['num_neg']+1)
            # (batch_size, num_neg+1)
            sim_scores = torch.bmm(q_outputs, p_outputs)
            # 각 배치의 몇번째 클래스가 정답인지 지정. 첫번째가 pos이므로 모두 0
            targets = torch.zeros(args['batch_size']).long()
            targets = targets.to(device)
            sim_scores = sim_scores.view(args['batch_size'], -1)
            sim_scores = F.log_softmax(sim_scores, dim=1)

            # positive는 가깝게, negative는 멀게
            loss = F.nll_loss(sim_scores, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            encoder.zero_grad()
            torch.cuda.empty_cache()

            if args['wandb'] == True:
                wandb.log({'Train/train_loss': loss.item(),
                          'epoch': epoch}, step=step)
                step += 1

            if i % args['report_step'] == 0:
                print(f"Loss: {loss.item()}")
        print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))

        acc = validation(encoder, device, val_dataloader, doc_id)
        print(f" {epoch}th epoch TOP 10 ACC : {acc}")
        
        # wandb 기록
        if args['wandb'] == True:
            wandb.log({
                        'Val/val_top10acc': acc,
                        'epoch':epoch}, step=step)
        if acc > best_acc:
            best_acc = acc
            encoder.save_pretrained(f"model/{args['name']}/ep{epoch}_acc{acc:.3f}")
      


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_train', default='config/config_train.yaml',
                        type=str, help='path of train configuration yaml file')

    pre_args = parser.parse_args()

    with open(pre_args.config_train) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    if args['wandb'] == True:
        wandb.login()
        wandb.init(project="Search", entity='gome', name=args['name'])
        wandb.config.update(args)

    train(args)

    # from tokenizers import Tokenizer
    # tokenizer_path = 'data/tokenizer/tokenizer-bpe.json'
    # tokenizer = Tokenizer.from_file(tokenizer_path)
    # dataset = make_val_dataset(args, tokenizer)
    # print(len(dataset[0]), len(dataset[0][0].shape))
    # print(dataset[0])
