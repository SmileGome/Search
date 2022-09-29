from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import json

# make formula.txt
def plus(text):
  text += "\n"
  return text

def make_formulafile(file_name):
  with open(file_name) as f:
    data = json.load(f)

  with open("data/tokenizer/formulas.txt", "w") as file:
    cnt = 0
    for doc in data:
      cnt += 1
      latexes = doc['latex']
      lst = list(map(plus, latexes))
      file.writelines(lst)
  print(f'done {cnt}th line formulas.txt !!!')

# train bpe tokenizer
def make_tokenizer(formulas_file, max_length = 100, vocab_size = 600):
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
    tokenizer.save("data/tokenizer/tokenizer-bpe.json")
    return tokenizer

if __name__=='__main__':
  file_name = 'data/clean_anno.json'
  make_formulafile(file_name)
  make_tokenizer('data/tokenizer/formulas.txt', max_length=100, vocab_size=600)
