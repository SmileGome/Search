# set seed
from dataset import make_final_dataset, split_train_test

import numpy as np
import random
import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from transformers import DPRConfig, DPRQuestionEncoder, TrainingArguments, get_linear_schedule_with_warmup, AdamW
from tqdm import tqdm
import wandb
import argparse
import yaml


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train(args):
    # seed
    seed_everything(args['seed'])

    # settings
    print("pytorch version: {}".format(torch.__version__))
    gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    print("GPU 사용 가능 여부: {}".format(gpu))
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    # load tokenizer
    tokenizer = Tokenizer.from_file(args['tokenizer_path'])

    # load dataset
    split_train_test(args['all_file_name'])
    train_dataset = make_final_dataset(args['train_file_name'], args['num_neg'], tokenizer)
    val_dataset = make_final_dataset(args['val_file_name'], args['num_neg'], tokenizer)
    test_dataset = make_final_dataset(args['test_file_name'], args['num_neg'], tokenizer)


    # make dataloader
    train_dataloader = DataLoader(
        train_dataset, batch_size=args['batch_size'], shuffle=True, drop_last=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args['batch_size'], shuffle=False, drop_last=True)

    # load model
    config = DPRConfig.from_pretrained(args['model_path'])
    q_encoder = DPRQuestionEncoder.from_pretrained(
        args['model_path'], config=config)
    p_encoder = DPRQuestionEncoder.from_pretrained(
        args['model_path'], config=config)
    q_encoder, p_encoder = q_encoder.to(device), p_encoder.to(device)

    def change_config(model, tokenizer):
        model.config.decoder_start_token_id = tokenizer.token_to_id("[CLS]")
        model.config.pad_token_id = tokenizer.token_to_id("[PAD]")
        model.config.vocab_size = args['vocab_size']

        # set beam search parameters
        model.config.eos_token_id = tokenizer.token_to_id("[SEP]")
        model.config.max_length = args['max_length_token']
        model.config.early_stopping = True
        model.config.no_repeat_ngram_size = 3
        model.config.length_penalty = 2.0
        model.config.num_beams = 4

    change_config(q_encoder, tokenizer)
    change_config(p_encoder, tokenizer)

    # start training
    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in p_encoder.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
        {'params': [p for n, p in p_encoder.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in q_encoder.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
        {'params': [p for n, p in q_encoder.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args['learning_rate'], eps=args['adam_epsilon'])
    t_total = len(
        train_dataloader) // args['gradient_accumulation_steps'] * args['num_epoch']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps=t_total)

    step = 0
    # loop over the dataset multiple times
    for epoch in range(args['num_epoch']):
        # train
        q_encoder.train()
        p_encoder.train()

        train_loss = 0.0
        for i, batch in enumerate(tqdm(train_dataloader)):
            batch = tuple(t.to(device) for t in batch)

            p_inputs = {'input_ids': batch[0].view(args['batch_size']*(args['num_neg']+1), -1), 
                        'token_type_ids': batch[1].view(args['batch_size']*(args['num_neg']+1), -1),
                        'attention_mask': batch[2].view(args['batch_size']*(args['num_neg']+1), -1)
                        }
            
            q_inputs = {'input_ids': batch[0],
                        'token_type_ids': batch[1],
                        'attention_mask': batch[2]
                        }      
                                
            p_outputs = p_encoder(**p_inputs).pooler_output  #(batch_size*(args['num_neg']+1), emb_dim)
            q_outputs = q_encoder(**q_inputs).pooler_output  #(batch_size*, emb_dim)
            q_outputs = torch.unsqueeze(q_outputs, 1)
            p_outputs = p_outputs.reshape(args['batch_size'], -1, args['num_neg']+1)
            sim_scores = torch.bmm(q_outputs, p_outputs) 
            targets = torch.zeros(args['batch_size']).long()    
            targets[0] = 1  
            targets = targets.to(device)
            sim_scores = sim_scores.view(args['batch_size'], -1)
            sim_scores = F.log_softmax(sim_scores, dim=1)

            # positive는 가깝게, negative는 멀게
            loss = F.nll_loss(sim_scores, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()
            q_encoder.zero_grad()
            p_encoder.zero_grad()
            torch.cuda.empty_cache()

            if args['wandb'] == True:
                wandb.log({'Train/train_loss': loss.item(),
                          'epoch': epoch}, step=step)
                step += 1

            if i % args['report_step'] == 0:
                print(f"Loss: {loss.item()}")

        print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))

        # validate
        print(f'Start {epoch}th evaluation!!!')
        q_encoder.eval()
        p_encoder.eval()

        val_loss = 0.0
        best_loss = int(1e9)

        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_dataloader)):
                batch = tuple(t.to(device) for t in batch)

                p_inputs = {'input_ids': batch[0].view(args['batch_size']*(args['num_neg']+1), -1),
                            'attention_mask': batch[1].view(args['batch_size']*(args['num_neg']+1), -1),
                            'token_type_ids': batch[2].view(args['batch_size']*(args['num_neg']+1), -1)
                            }

                q_inputs = {'input_ids': batch[3],
                            'attention_mask': batch[4],
                            'token_type_ids': batch[5]}

                # (batch_size*(args['num_neg']+1), emb_dim) # 30, 768
                p_outputs = p_encoder(**p_inputs).pooler_output
                # (batch_size*, emb_dim)
                q_outputs = q_encoder(**q_inputs).pooler_output
                q_outputs = torch.unsqueeze(q_outputs, 1)
                p_outputs = p_outputs.reshape(
                    args['batch_size'], -1, args['num_neg']+1)
                sim_scores = torch.bmm(q_outputs, p_outputs)

                # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)
                # batch내의 모든 쿼리와 passage간의 유사도 구하기(matmul)

                targets = torch.zeros(args['batch_size']).long()
                targets = targets.to(device)

                sim_scores = sim_scores.view(args['batch_size'], -1)
                sim_scores = F.log_softmax(sim_scores, dim=1)

                # positive는 가깝게, negative는 멀게
                loss = F.nll_loss(sim_scores, targets)

                torch.cuda.empty_cache()

                val_loss += loss.item()

        epoch_loss = val_loss / len(val_dataloader)

        print(f"{epoch}th epoch Val LOSS:{epoch_loss}")
        if args['wandb'] == True:
            wandb.log({
                'Val/val_loss': epoch_loss,
                'epoch': epoch}, step=step)

        if epoch_loss < best_loss:
            p_encoder.save_pretrained(
                f"model/{args['name']}/p_encoder/ep{epoch}_l{epoch_loss:.3f}")
            q_encoder.save_pretrained(
                f"model/{args['name']}/q_encoder/ep{epoch}_l{epoch_loss:.3f}")


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
