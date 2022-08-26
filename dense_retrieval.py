import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import pandas as pd

import json
import random
from tqdm.auto import tqdm
from pprint import pprint
import wandb
import argparse
import os 
import time
from datasets import load_from_disk, load_dataset
from typing import List, NoReturn, Optional, Tuple, Union
from datasets import (
    Dataset,
    DatasetDict,
)
from utils.arguments import DataTrainingArguments, ModelArguments
from contextlib import contextmanager
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
    HfArgumentParser,
)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

# 난수 고정
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)
    
set_seed(42) # magic number :)

class DenseRetrieval:
    def __init__(self, args, dataset, tokenizer, p_encoder, q_encoder, num_neg=2, mode='train', bm25=False):

        self.args = args[0] # TrainingArguments
        self.additional_args = args[1] # ModelArguments
        self.dataset = dataset
        self.num_neg = num_neg

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.bm25 = False
        self.q_embs = None
        self.p_embs = None
        self.mode = mode

        if bm25 == 'True':      # bm25
            corpus = np.array(list(set([example for example in self.dataset['context']])))

            tokenized_corpus = [self.split_space(doc) for doc in corpus]    # 띄어쓰기로 토큰화 (우리는 우리 토크나이저로 변경하면 될 듯)
            self.bm25 = BM25Okapi(tokenized_corpus)

        if mode == 'train':
            self.prepare_in_batch_negative()

        else:
            # inference를 위한 passages 임베딩 시 수행 (default=train)
            if mode == 'inference':
                self.get_dense_embedding(dataset)
            else:
                self.get_dense_embedding(dataset)
                self.compute_topk()

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        assert self.p_embs is not None, "get_dense_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i + 1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": ('\n' + "="*150 + '\n').join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas
    
    # query, passage embedding 수행하여 변수에 저장
    def get_dense_embedding(self, dataset):

        args = self.args
        p_encoder = self.p_encoder
        q_encoder = self.q_encoder
        tokenizer = self.tokenizer
        BATCHSIZE = 32

        if self.mode=='inference':
            passages, questions = dataset[0], dataset[1]

            passages_list = []
            for idx, p in enumerate(passages):
                passages[str(idx)]['text']
                passages_list.append(passages[str(idx)]['text'])

            print('[passages num]:', len(passages_list))

            p_seqs = tokenizer(passages_list, padding="max_length", truncation=True, return_tensors='pt')

            passage_dataset = TensorDataset(
                p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
            )

            self.passage_dataloader = DataLoader(passage_dataset, batch_size=BATCHSIZE)
            
            with torch.no_grad():
                p_encoder.eval()
                p_embs=[]

                with tqdm(self.passage_dataloader, unit="batch") as tepoch:
                    for idx, batch in enumerate(tepoch):

                        p_inputs = {
                            'input_ids': batch[0].to(args.device),
                            'attention_mask': batch[1].to(args.device),
                            'token_type_ids': batch[2].to(args.device)
                        }
                
                        p_outputs = self.p_encoder(**p_inputs) # (batch_size*(num_neg+1), emb_dim)
                        p_embs.append(p_outputs)

            self.p_embs = torch.concat(p_embs, dim=0)
            self.contexts = passages_list

        else:
            # 2. (Question, Passage) 데이터셋 만들어주기
            q_seqs = tokenizer(dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
            p_seqs = tokenizer(dataset['context'], padding="max_length", truncation=True, return_tensors='pt')

            print(q_seqs['input_ids'].shape, p_seqs['input_ids'].shape)

            train_dataset = TensorDataset(
                p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
                q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
            )

            passage_dataset = TensorDataset(
                p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
            )

            self.passage_dataloader = DataLoader(passage_dataset, batch_size=BATCHSIZE)

            dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE)

            with torch.no_grad():
                p_encoder.eval()
                q_encoder.eval()
                p_embs, q_embs = [], []

                with tqdm(dataloader, unit="batch") as tepoch:
                    for idx, batch in enumerate(tepoch):

                        p_inputs = {
                            'input_ids': batch[0].to(args.device),
                            'attention_mask': batch[1].to(args.device),
                            'token_type_ids': batch[2].to(args.device)
                        }
                
                        q_inputs = {
                            'input_ids': batch[3].to(args.device),
                            'attention_mask': batch[4].to(args.device),
                            'token_type_ids': batch[5].to(args.device)
                        }
                
                        p_outputs = self.p_encoder(**p_inputs) # (batch_size*(num_neg+1), emb_dim)
                        p_embs.append(p_outputs)
                        q_outputs = self.q_encoder(**q_inputs) # (batch_size*, emb_dim)
                        q_embs.append(q_outputs)

            self.q_embs = torch.concat(q_embs, dim=0)
            self.p_embs = torch.concat(p_embs, dim=0)

    def compute_topk(self):
        dataset_len = self.q_embs.size(0)
        top1, top20, top100 = 0, 0, 0

        # 쿼리 하나씩 받아오면서 계산하기 
        for idx in range(dataset_len):
            q_emb = self.q_embs[idx,:]
            
            dot_prod_scores = torch.matmul(q_emb, torch.transpose(self.p_embs, 0, 1))
            rank = torch.argsort(dot_prod_scores, dim=0, descending=True).squeeze()

            if idx in rank[:100]:
                top100 += 1
            if idx in rank[:20]:
                top20 += 1
            if idx == rank[0]:
                top1 += 1

        top1_acc = top1/dataset_len
        top20_acc = top20/dataset_len
        top100_acc = top100/dataset_len

        print('[Top-1 acc]', top1_acc, ' | ', '[Top-20 acc]', top20_acc, ' | ', '[Top-100 acc]', top100_acc)

    def split_space(self, sent):
        return sent.split(" ")

    def BM25(self, query, corpus):

        tokenized_query = self.split_space(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_n_passages = self.bm25.get_top_n(tokenized_query, corpus, n=10)
        
        return top_n_passages

    def prepare_in_batch_negative(self, dataset=None, tokenizer=None):

        dataset = self.dataset
        tokenizer = self.tokenizer

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.        
        corpus = np.array(list(set([example for example in dataset['context']])))

        # tokenized_corpus = [tokenizer(doc) for doc in corpus]
        # self.bm25 = BM25Okapi(tokenized_corpus)

        p_with_neg = []
        
        if self.bm25:
            num = self.num_neg + 1 + self.additional_args.bm_num
        else:
            num = self.num_neg + 1

        print("prepare_in_batch_negative")
        with tqdm(dataset['context'], unit="batch") as tepoch:
            for idx, c in enumerate(tepoch):
                while True:
                    neg_idxs = np.random.randint(len(corpus), size=self.num_neg)

                    if not c in corpus[neg_idxs]:
                        p_neg = corpus[neg_idxs]

                        p_with_neg.append(c)
                        p_with_neg.extend(p_neg)

                        # BM25로 질문과 유사도 높은 지문 negative sample로 추가 (--bm_num으로 몇개 추가할건지 설정가능)
                        if self.bm25:
                            cnt = 0
                            top_n_passages = self.BM25(dataset['question'][idx], corpus)
                            
                            for p in top_n_passages:
                                if p != c:
                                    p_with_neg.append(p)
                                    cnt += 1
                                if cnt == self.additional_args.bm_num:
                                    break

                            # BM25로 질문과 유사도 높은 지문 negative sample로 추가 (성능 별로 좋지 않았음, 전처리 오래 걸림!)
                            # top_n_passages = self.BM25(dataset['context'][idx], corpus)
                            
                            # for p in top_n_passages:
                            #     if p != c:
                            #         p_with_neg.append(p)
                            #         break
                        break
                    

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt')

        max_len = p_seqs['input_ids'].size(-1)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num, max_len)

        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size)


    def train(self, args=None):

        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // self.additional_args.dpr_gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        # Start training!
        global_step = 0
        global_loss = 100

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")

        if self.bm25:
            num = self.num_neg + 1 + self.additional_args.bm_num
        else:
            num = self.num_neg + 1

        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for idx, batch in enumerate(tepoch):

                    self.p_encoder.train()
                    self.q_encoder.train()
                    
                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        'input_ids': batch[0].view(batch_size * (num), -1).to(args.device),
                        'attention_mask': batch[1].view(batch_size * (num), -1).to(args.device),
                        'token_type_ids': batch[2].view(batch_size * (num), -1).to(args.device)
                    }
            
                    q_inputs = {
                        'input_ids': batch[3].to(args.device),
                        'attention_mask': batch[4].to(args.device),
                        'token_type_ids': batch[5].to(args.device)
                    }
            
                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, num, -1)
     
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  #(batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)

                    if self.additional_args.wandb=='True':
                        wandb.log({"loss": loss})

                    # loss 값 제일 낮을 때 encoder 저장
                    if loss < global_loss:
                        global_loss=loss
                        # model save
                        
                        os.makedirs(self.additional_args.encoder_save_dir, exist_ok=True)
                        p_encoder_save_path = os.path.join(self.additional_args.encoder_save_dir, f"p_encoder")
                        q_encoder_save_path = os.path.join(self.additional_args.encoder_save_dir, f"q_encoder")
                        
                        self.p_encoder.save_pretrained(p_encoder_save_path)
                        self.q_encoder.save_pretrained(q_encoder_save_path)

                    tepoch.set_postfix(loss=f'{str(loss.item())}')

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        args = self.args
        p_encoder = self.p_encoder
        q_encoder = self.q_encoder

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to(args.device)
            q_emb = q_encoder(**q_seqs_val).to('cpu')  

        p_embs = torch.tensor(self.p_embs, device='cpu').squeeze()
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1)).squeeze()
        rank = torch.argsort(dot_prod_scores, descending=True) 

        doc_score = dot_prod_scores[rank].tolist()[:k]
        doc_indices = rank.tolist()[:k]

        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        args = self.args
        p_encoder = self.p_encoder
        q_encoder = self.q_encoder

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors='pt').to(args.device)
            q_embs = q_encoder(**q_seqs_val).to('cpu')  

        p_embs = torch.tensor(self.p_embs, device='cpu').squeeze()
        dot_prod_scores = torch.matmul(q_embs, torch.transpose(p_embs, 0, 1)).squeeze()

        rank = torch.argsort(dot_prod_scores, descending=True) 

        tensor_stack = []
        for i in range(dot_prod_scores.size(0)):
            tensor_stack.append(dot_prod_scores[i, rank[i, :k]])

        doc_scores = torch.stack(tensor_stack, dim=0).tolist()
        doc_indices = rank[:, :k].tolist()
        
        return doc_scores, doc_indices
  
class BertEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()
      
    def forward(
            self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 
  
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        return pooled_output

def main(args):

    # arguments.py 참고 (--epochs --batch_size --num_neg --save_dir --report_name --project_name --wandb --test_query --bm25 --bm_num --dataset --topk)
    # [example] python dense_retrieval.py --report_name HY-BERT_baseline_wiki_BM25_ex3 --bm25 True --num_neg 4 --bm_num 2 --dataset wiki --wandb True
    
    if args.wandb == 'True':
        wandb.init(project=args.project_name, entity="salt-bread", name=args.report_name)
    
    print(args)

    # 대회 데이터셋 불러오기
    if args.dataset == 'wiki':
        dataset_train = load_from_disk("../data/train_dataset/")
        train_dataset = dataset_train['train']

    # korQuad 불러오기
    if args.dataset == 'squad_kor_v1':
        train_dataset = load_dataset("squad_kor_v1")['train']

    train_args = TrainingArguments(
        output_dir=args.encoder_save_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size, # 아슬아슬합니다. 작게 쓰세요 !
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
    )

    # Train 
    model_checkpoint = 'klue/bert-base'

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(train_args.device)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(train_args.device)

    retriever = DenseRetrieval(args=[train_args, args], dataset=train_dataset, bm25=args.bm25, num_neg=args.num_neg, tokenizer=tokenizer, p_encoder=p_encoder, q_encoder=q_encoder)
    retriever.train()

    if args.test_query == 'True':
    
        model_checkpoint = 'klue/bert-base'
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        p_encoder_path = os.path.join(args.encoder_save_dir, f"p_encoder")
        q_encoder_path = os.path.join(args.encoder_save_dir, f"q_encoder")
        
        p_encoder = BertEncoder.from_pretrained(p_encoder_path).to(train_args.device)
        q_encoder = BertEncoder.from_pretrained(q_encoder_path).to(train_args.device)

        # squad_kor_v1 데이터셋
        # dataset = load_dataset("squad_kor_v1")['train']

        # 대회 validation set
        dataset = load_from_disk("../data/train_dataset/")
        dataset = dataset['validation']

        retriever = DenseRetrieval(args=[train_args, args], dataset=dataset, bm25=args.bm25, num_neg=args.num_neg, tokenizer=tokenizer, p_encoder=p_encoder, q_encoder=q_encoder, mode='test')

        # # 단일 쿼리 테스트 (str)
        index = 0
        doc_scores, doc_indices = retriever.get_relevant_doc(query=dataset[index]['question'], k=args.topk)

        print(f"[Search Query] {dataset[index]['question']}\n")
        print(f"[Passage] {dataset[index]['context']}\n")

        for i, idx in enumerate(doc_indices):
            print(f"Top-{i + 1}th Passage (Score {doc_scores[i]})")
            pprint(retriever.dataset['context'][idx])

        # # 다중 쿼리 테스트 (List)
        # doc_scores, doc_indices = retriever.get_relevant_doc_bulk(queries=[dataset[index]['question'],dataset[index+1]['question'],dataset[index+2]['question']] ,k=args.topk)

        # for i in range(len(doc_indices)):
        #     for j, idx in enumerate(doc_indices[i]):
        #         pprint(retriever.dataset['question'][idx])
        #         print(f"Top-{j + 1}th Passage (Score {doc_scores[i][j]})")
        #         pprint(retriever.dataset['context'][idx])
        #     print("---------------------------------------------------------------------")

if __name__ == '__main__':
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments)
    )
    args, _ = parser.parse_args_into_dataclasses()
    main(args)
