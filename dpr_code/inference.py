from tokenizers import Tokenizer
import torch
from transformers import DPRConfig, DPRQuestionEncoder
import pickle


# input string 1개 output embedding(384) 1개
def embedding(input_str, tokenizer_path, model_path):
  device = torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")
  tokenizer = Tokenizer.from_file(tokenizer_path)
  input_token = tokenizer.encode(input_str)
  input = {}
  input['input_ids'] = torch.tensor(input_token.ids).unsqueeze(dim=0).to(device)
  input['token_type_ids'] = torch.tensor(input_token.type_ids).unsqueeze(dim=0).to(device)
  input['attention_mask'] = torch.tensor(input_token.attention_mask).unsqueeze(dim=0).to(device)

  # load model
  config = DPRConfig.from_pretrained(model_path)
  encoder = DPRQuestionEncoder.from_pretrained(
      model_path, config=config,
      ignore_mismatched_sizes=True
  )
  encoder = encoder.to(device)
  encoder.eval()
  with torch.no_grad():
    output = encoder(**input).pooler_output.tolist()
  return output


if __name__=="__main__":
  with open('data/result/backend_sample/backend_sample_input.pkl', 'rb') as f:
    input_str = pickle.load(f)
  tokenizer_path = 'data/tokenizer/tokenizer-bpe.json'
  model_path = 'model\small_pool_384_shortval\ep0_acc0.847'
  emb = embedding(input_str, tokenizer_path, model_path)
  print(type(emb), len(emb))
  # with open('data/backend_sample_output.pkl', 'wb') as f:
  #   pickle.dump(emb, f)