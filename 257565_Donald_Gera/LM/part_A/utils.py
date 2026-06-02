from functools import partial

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

from transformers import AutoTokenizer


# Read one PTB split and append an explicit end-of-sentence marker to every line.
def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                output.append(line + " " + eos_token)
            else:
                output.append(eos_token)
    return output


# Load the fixed Penn Treebank train, validation, and test files expected by this part.
def load_dataset(eos_token="<eos>"):
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt", eos_token=eos_token)
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt", eos_token=eos_token)
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt", eos_token=eos_token)
    return train_raw, dev_raw, test_raw


# Minimal Dataset wrapper: tokenization is intentionally deferred to collation time.
class PennTreeBank(data.Dataset):
    def __init__(self, corpus):
        self.sents = [sent for sent in corpus]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]


def load_tokenizer(tokenizer_name="openai-community/gpt2"):
    # GPT-2 has no native padding token, so reuse EOS for padded batch positions.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# Expose the tokenizer vocabulary for dataset statistics printed by main.py.
def build_vocabulary(tokenizer):
    return tokenizer.get_vocab()


def collate_fn(batch, tokenizer, device):
    # Tokenize and pad dynamically to the longest sentence in the current batch.
    tokenized = tokenizer(batch, padding=True, return_tensors="pt")

    # Shift by one position: each input token is trained to predict the next token.
    input_ids = tokenized.input_ids[:, :-1].detach().clone().to(device)
    labels = tokenized.input_ids[:, 1:].detach().clone().to(device)
    # Count only non-padding labels so losses can be averaged correctly across batches.
    n_tokens = torch.sum(labels != tokenizer.pad_token_id)

    return input_ids, labels, n_tokens


# Wrap the raw text splits in Dataset objects consumed by the dataloaders.
def make_datasets(eos_token="<eos>"):
    train_raw, dev_raw, test_raw = load_dataset(eos_token=eos_token)

    train_dataset = PennTreeBank(train_raw)
    dev_dataset = PennTreeBank(dev_raw)
    test_dataset = PennTreeBank(test_raw)

    return train_dataset, dev_dataset, test_dataset


def make_dataloaders(tokenizer, device, train_batch_size=8, eval_batch_size=16, seed=42):
    # Build reproducible shuffled training batches and deterministic evaluation batches.
    train_dataset, dev_dataset, test_dataset = make_datasets()

    # DataLoader uses its own generator for reproducible training-set shuffling.
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Bind shared tokenizer/device arguments before passing the collator to DataLoader.
    collate = partial(collate_fn, tokenizer=tokenizer, device=device)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate, generator=generator)
    dev_loader = DataLoader(dev_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate)

    return train_loader, dev_loader, test_loader
