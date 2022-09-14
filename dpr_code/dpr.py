import json
import pandas as pd
from os.path import join
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from transformers import DPRConfig, DPRQuestionEncoder, TrainingArguments, get_linear_schedule_with_warmup, AdamW
from tqdm import tqdm


# make formula.txt
def plus(text):
  text += "\n"
  return text

def make_formulafile(file_name):
  with open(file_name) as f:
    data = json.load(f)

  with open("data/formulas.txt", "w") as file:
    cnt = 0
    for doc in data['latex_anno']:
      cnt += 1
      latexes = doc['latex']
      lst = list(map(plus, latexes))
      file.writelines(lst)
  print('done formula.txt !!!')

# train bpe tokenizer
def make_tokenizer(formulas_file, max_length = None, vocab_size = None):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.enable_padding(length=max_length)
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                        vocab_size=vocab_size,
                        show_progress=True,
                        )

    files = [formulas_file]
    tokenizer.train(files, trainer)
    tokenizer.save("data/tokenizer-bpe.json")
    return tokenizer

# anno.json -> dataset format
def make_dataset(file_name):
  with open(file_name) as f:
    data = json.load(f)

  question = []
  context = []

  for doc in data['latex_anno']:
    latexes = doc['latex']
    if len(latexes)==1:
      continue
    leng = set(range(len(latexes)))
    for idx, l in enumerate(latexes):
      question.append(l)
      index = np.array(list(leng-{idx}))
      context.append(' '.join(list(np.array(latexes)[index])))
  return question, context    

##################################################################

tokenizer = Tokenizer.from_file("data/tokenizer-wordlevel.json")
question, context = make_dataset('data/anno.json')
q_seqs = tokenizer.encode_batch(question)
p_seqs = tokenizer.encode_batch(context)

q_dataset = {}
p_dataset = {}

for idx, (q_encoding, p_encoding) in enumerate(zip(q_seqs, p_seqs)):
  if idx==0:
    q_dataset['input_ids'] = torch.tensor(q_encoding.ids).view(1,-1)
    q_dataset['token_type_ids'] = torch.tensor(q_encoding.type_ids).view(1,-1)
    q_dataset['attention_mask'] = torch.tensor(q_encoding.attention_mask).view(1,-1)
    
    p_dataset['input_ids'] = torch.tensor(p_encoding.ids).view(1,-1)
    p_dataset['token_type_ids'] = torch.tensor(p_encoding.type_ids).view(1,-1)
    p_dataset['attention_mask'] = torch.tensor(p_encoding.attention_mask).view(1,-1)

  else:
    q_dataset['input_ids'] = torch.cat([q_dataset['input_ids'], torch.tensor(q_encoding.ids).view(1,-1)], dim=0)
    q_dataset['token_type_ids'] = torch.cat([q_dataset['token_type_ids'], torch.tensor(q_encoding.type_ids).view(1,-1)], dim=0)
    q_dataset['attention_mask'] = torch.cat([q_dataset['attention_mask'], torch.tensor(q_encoding.attention_mask).view(1,-1)], dim=0)
    
    p_dataset['input_ids'] = torch.cat([p_dataset['input_ids'], torch.tensor(p_encoding.ids).view(1,-1)], dim=0)
    p_dataset['token_type_ids'] = torch.cat([p_dataset['token_type_ids'], torch.tensor(p_encoding.type_ids).view(1,-1)], dim=0)
    p_dataset['attention_mask'] = torch.cat([p_dataset['attention_mask'], torch.tensor(p_encoding.attention_mask).view(1,-1)], dim=0)

train_dataset = TensorDataset(p_dataset['input_ids'], p_dataset['attention_mask'], p_dataset['token_type_ids'], 
                        q_dataset['input_ids'], q_dataset['attention_mask'], q_dataset['token_type_ids'])

model_path = 'facebook/dpr-question_encoder-single-nq-base'

DPR_config = DPRConfig.from_pretrained(model_path)
q_encoder = DPRQuestionEncoder.from_pretrained(model_path, config=DPR_config)
p_encoder = DPRQuestionEncoder.from_pretrained(model_path, config=DPR_config)

vocab_size = 600
max_length_token = 100
  
def change_config(model, tokenizer):
  # set special tokens used for creating the decoder_input_ids from the labels
  model.config.decoder_start_token_id = tokenizer.token_to_id("[CLS]")
  model.config.pad_token_id = tokenizer.token_to_id("[PAD]")
  # make sure vocab size is set correctly
  model.config.vocab_size = vocab_size
  # model.max_position_embeddings = max_length_token

  # set beam search parameters
  model.config.eos_token_id = tokenizer.token_to_id("[SEP]")
  model.config.max_length = max_length_token
  model.config.early_stopping = True
  model.config.no_repeat_ngram_size = 3
  model.config.length_penalty = 2.0
  model.config.num_beams = 4

  change_config(q_encoder, tokenizer)
  change_config(p_encoder, tokenizer)
  
  args = TrainingArguments(
    output_dir="dense_retireval",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    weight_decay=0.01
    )

  if torch.cuda.is_available():
    p_encoder.cuda()
    q_encoder.cuda()

def train(args, dataset, p_model, q_model):
  
  # Dataloader
  # train_sampler = RandomSampler(dataset)
  train_dataloader = DataLoader(dataset, batch_size=args.per_device_train_batch_size, shuffle=True, drop_last=True)

  # Optimizer
  no_decay = ['bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
        {'params': [p for n, p in p_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in p_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in q_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in q_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
  t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

  # Start training!
  global_step = 0
  
  p_model.zero_grad()
  q_model.zero_grad()
  torch.cuda.empty_cache()
  
  train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")

  for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")

    for step, batch in enumerate(epoch_iterator):
      q_encoder.train()
      p_encoder.train()
      
      if torch.cuda.is_available():
        batch = tuple(t.cuda() for t in batch)

      p_inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2]
                  }
      
      q_inputs = {'input_ids': batch[3],
                  'attention_mask': batch[4],
                  'token_type_ids': batch[5]}

      
      p_outputs = p_model(**p_inputs).pooler_output  #(batch_size*(num_neg+1), emb_dim)
      q_outputs = q_model(**q_inputs).pooler_output  #(batch_size*, emb_dim)

      sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1)) 
     
      # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)
      # batch내의 모든 쿼리와 passage간의 유사도 구하기(matmul)

      # target: position of positive samples = diagonal element 
      # 대각원소의 값이 커지는 방향으로 학습해야함
      targets = torch.arange(0, args.per_device_train_batch_size).long()
      

      if torch.cuda.is_available():
        targets = targets.to('cuda')

      sim_scores = F.log_softmax(sim_scores, dim=1)

      loss = F.nll_loss(sim_scores, targets) # positive는 가깝게, negative는 멀게
      print(loss.item())

      loss.backward()
      optimizer.step()
      scheduler.step()
      q_model.zero_grad()
      p_model.zero_grad()
      global_step += 1
      
      torch.cuda.empty_cache()

  return p_model, q_model

  
if __name__=='__main__':
  p_encoder, q_encoder = train(args, train_dataset, p_encoder, q_encoder)