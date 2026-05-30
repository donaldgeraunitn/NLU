import json
import os
from collections import Counter
from functools import partial

import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

IGNORE_INDEX = -100


def load_data(path):
    with open(path, encoding="utf-8") as f:
        return json.loads(f.read())


def load_dataset(dataset_dir="dataset/ATIS", dev_size=0.10, seed=42):
    tmp_train_raw = load_data(os.path.join(dataset_dir, "train.json"))
    test_raw = load_data(os.path.join(dataset_dir, "test.json"))

    intents = [x["intent"] for x in tmp_train_raw]
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])

    train_raw, dev_raw, _, _ = train_test_split(
        inputs,
        labels,
        test_size=dev_size,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )

    train_raw.extend(mini_train)
    return train_raw, dev_raw, test_raw


class Lang:
    def __init__(self, intents, slots):
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents)

        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def lab2id(self, elements):
        vocab = {}
        for elem in elements:
            if elem not in vocab:
                vocab[elem] = len(vocab)
        return vocab

    def to_dict(self):
        return {
            "slot2id": self.slot2id,
            "intent2id": self.intent2id,
        }


def build_lang(train_raw, dev_raw, test_raw):
    corpus = train_raw + dev_raw + test_raw
    slots = sorted(set(sum([line["slots"].split() for line in corpus], [])))
    intents = sorted(set([line["intent"] for line in corpus]))

    return Lang(intents, slots)


class IntentsAndSlots(data.Dataset):
    def __init__(self, dataset, lang):
        self.utterances = []
        self.intents = []
        self.slots = []

        for x in dataset:
            words = x["utterance"].split()
            slots = x["slots"].split()

            if len(words) != len(slots):
                raise ValueError(
                    "The number of words and slot labels must match: "
                    f"{x['utterance']}"
                )

            self.utterances.append(words)
            self.slots.append(slots)
            self.intents.append(lang.intent2id[x["intent"]])

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        return {
            "utterance": self.utterances[idx],
            "slots": self.slots[idx],
            "intent": self.intents[idx],
        }


def make_datasets(dataset_dir="dataset/ATIS", dev_size=0.10, seed=42):
    train_raw, dev_raw, test_raw = load_dataset(dataset_dir=dataset_dir, dev_size=dev_size, seed=seed,)
    lang = build_lang(train_raw, dev_raw, test_raw)

    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    return train_dataset, dev_dataset, test_dataset, lang, train_raw, dev_raw, test_raw


def make_tokenizer(model_type, model_name):
    if model_type == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            add_prefix_space=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
    elif model_type == "bert":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required to use word_ids().")

    return tokenizer


def _align_slot_labels(slot_labels, word_ids, slot2id, model):
    if strategy not in {"first", "last"}:
        raise ValueError(f"Unsupported alignment strategy: {strategy}")

    aligned_labels = []

    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            aligned_labels.append(IGNORE_INDEX)
            continue

        if strategy == "first":
            selected = idx == 0 or word_ids[idx - 1] != word_id
        else:
            selected = idx == len(word_ids) - 1 or word_ids[idx + 1] != word_id

        if selected:
            aligned_labels.append(slot2id[slot_labels[word_id]])
        else:
            aligned_labels.append(IGNORE_INDEX)

    return aligned_labels


def _truncate_words(words, word_ids):
    valid_word_ids = [word_id for word_id in word_ids if word_id is not None]
    n_words = 0 if not valid_word_ids else max(valid_word_ids) + 1
    return words[:n_words]


def _collate_bert(batch, tokenizer, lang, device, max_length):
    utterances = [sample["utterance"] for sample in batch]
    encoded = tokenizer(utterances, is_split_into_words=True, padding=True, truncation=True, max_length=max_length, return_tensors="pt", )

    y_slots = []
    words = []
    slots_len = []

    for idx, sample in enumerate(batch):
        word_ids = encoded.word_ids(batch_index=idx)
        aligned_labels = _align_slot_labels(sample["slots"], word_ids, lang.slot2id, strategy="first", )
        truncated_words = _truncate_words(sample["utterance"], word_ids)

        y_slots.append(aligned_labels)
        words.append(truncated_words)
        slots_len.append(len(truncated_words))

    return {
        "utterances": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "intents": torch.LongTensor([sample["intent"] for sample in batch]).to(device),
        "y_slots": torch.LongTensor(y_slots).to(device),
        "slots_len": torch.LongTensor(slots_len).to(device),
        "words": words,
    }


def _collate_gpt2(batch, tokenizer, lang, device, max_length):
    encoded_batch = []
    max_batch_length = 0

    for sample in batch:
        encoded = tokenizer(
            sample["utterance"],
            is_split_into_words=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - 1,
        )

        input_ids = encoded["input_ids"] + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        word_ids = encoded.word_ids() + [None]

        aligned_labels = _align_slot_labels( sample["slots"], word_ids, lang.slot2id, strategy="last", )
        truncated_words = _truncate_words(sample["utterance"], word_ids)

        encoded_batch.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "y_slots": aligned_labels,
                "words": truncated_words,
            }
        )
        max_batch_length = max(max_batch_length, len(input_ids))

    utterances = torch.full( (len(batch), max_batch_length), tokenizer.pad_token_id, dtype=torch.long, )
    attention_mask = torch.zeros((len(batch), max_batch_length), dtype=torch.long)
    y_slots = torch.full(
        (len(batch), max_batch_length),
        IGNORE_INDEX,
        dtype=torch.long,
    )

    for idx, encoded in enumerate(encoded_batch):
        length = len(encoded["input_ids"])
        utterances[idx, :length] = torch.LongTensor(encoded["input_ids"])
        attention_mask[idx, :length] = torch.LongTensor(encoded["attention_mask"])
        y_slots[idx, :length] = torch.LongTensor(encoded["y_slots"])

    words = [encoded["words"] for encoded in encoded_batch]

    return {
        "utterances": utterances.to(device),
        "attention_mask": attention_mask.to(device),
        "intents": torch.LongTensor([sample["intent"] for sample in batch]).to(device),
        "y_slots": y_slots.to(device),
        "slots_len": torch.LongTensor([len(item) for item in words]).to(device),
        "words": words,
    }


def collate_fn(batch, tokenizer, lang, model_type, device, max_length=50):
    if model_type == "bert":
        return _collate_bert(batch, tokenizer, lang, device, max_length)
    if model_type == "gpt2":
        return _collate_gpt2(batch, tokenizer, lang, device, max_length)
    raise ValueError(f"Unsupported model type: {model_type}")


def make_dataloaders(device, model_type, model_name, dataset_dir="dataset/ATIS", train_batch_size=32, eval_batch_size=64, seed=42, data_seed=None, dev_size=0.10, max_length=50):
    if data_seed is None:
        data_seed = seed

    train_dataset, dev_dataset, test_dataset, lang, train_raw, dev_raw, test_raw = make_datasets(dataset_dir=dataset_dir, dev_size=dev_size, seed=data_seed)
    tokenizer = make_tokenizer(model_type, model_name)

    generator = torch.Generator()
    generator.manual_seed(seed)

    collate = partial(collate_fn, tokenizer=tokenizer, lang=lang, model_type=model_type, device=device, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate, generator=generator)
    dev_loader = DataLoader(dev_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate)

    return train_loader, dev_loader, test_loader, lang, tokenizer, train_raw, dev_raw, test_raw
