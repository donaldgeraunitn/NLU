from functools import partial

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

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


def load_dataset(eos_token="<eos>"):
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt", eos_token=eos_token)
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt", eos_token=eos_token)
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt", eos_token=eos_token)
    return train_raw, dev_raw, test_raw


class PennTreeBank(data.Dataset):
    def __init__(self, corpus):
        self.sents = [sent for sent in corpus]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]


def load_tokenizer(tokenizer_name="openai-community/gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_vocabulary(tokenizer):
    return tokenizer.get_vocab()


def collate_fn(batch, tokenizer, device):
    tokenized = tokenizer(batch, padding=True, return_tensors="pt")

    # GPT2LMHeadModel shifts the labels internally. Unlike Part 1.A, inputs and
    # labels must therefore not be shifted manually in the collate function.
    input_ids = tokenized.input_ids.detach().clone().to(device)
    labels = input_ids.detach().clone()
    labels[labels == tokenizer.pad_token_id] = -100

    # GPT2LMHeadModel ignores the first label after its internal shift.
    n_tokens = torch.sum(labels[:, 1:] != -100)

    return input_ids, labels, n_tokens


def make_datasets(eos_token="<eos>"):
    train_raw, dev_raw, test_raw = load_dataset(eos_token=eos_token)

    train_dataset = PennTreeBank(train_raw)
    dev_dataset = PennTreeBank(dev_raw)
    test_dataset = PennTreeBank(test_raw)

    return train_dataset, dev_dataset, test_dataset


def make_dataloaders(
    tokenizer,
    device,
    train_batch_size=8,
    eval_batch_size=16,
    seed=42,
):
    train_dataset, dev_dataset, test_dataset = make_datasets()

    generator = torch.Generator()
    generator.manual_seed(seed)

    collate = partial(collate_fn, tokenizer=tokenizer, device=device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    return train_loader, dev_loader, test_loader
