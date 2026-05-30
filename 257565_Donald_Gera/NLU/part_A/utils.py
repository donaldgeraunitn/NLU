import json
import os
from collections import Counter
from functools import partial

import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

PAD_TOKEN = 0


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
    def __init__(self, words, intents, slots, cutoff=0, cls=True):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True, cls=cls)
        self.slot2id = self.lab2id(slots, cls=cls)
        self.intent2id = self.lab2id(intents, pad=False, cls=False)

        self.id2word = {v: k for k, v in self.word2id.items()}
        # CLS has the same id as PAD and must not overwrite PAD in id2slot.
        self.id2slot = {v: k for k, v in self.slot2id.items() if not cls or k != "cls"}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def w2id(self, elements, cutoff=None, unk=True, cls=True):
        vocab = {"pad": PAD_TOKEN}
        if unk:
            vocab["unk"] = len(vocab)
        if cls:
            vocab["cls"] = len(vocab)

        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab

    def lab2id(self, elements, pad=True, cls=True):
        vocab = {}
        if pad:
            vocab["pad"] = PAD_TOKEN
        for elem in elements:
            if elem not in vocab:
                vocab[elem] = len(vocab)
        if cls:
            vocab["cls"] = PAD_TOKEN
        return vocab

    def to_dict(self):
        return {
            "word2id": self.word2id,
            "slot2id": self.slot2id,
            "intent2id": self.intent2id,
        }


def build_lang(train_raw, dev_raw, test_raw, cutoff=0):
    words = sum([x["utterance"].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw

    slots = sorted(set(sum([line["slots"].split() for line in corpus], [])))
    intents = sorted(set([line["intent"] for line in corpus]))

    return Lang(words, intents, slots, cutoff=cutoff)


class IntentsAndSlots(data.Dataset):
    def __init__(self, dataset, lang, unk="unk", cls="cls", add_cls=True):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        self.cls = cls
        self.add_cls = add_cls

        for x in dataset:
            self.utterances.append(x["utterance"])
            self.slots.append(x["slots"])
            self.intents.append(x["intent"])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        return {
            "utterance": torch.LongTensor(self.utt_ids[idx]),
            "slots": torch.LongTensor(self.slot_ids[idx]),
            "intent": self.intent_ids[idx],
        }

    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, data, mapper):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                tmp_seq.append(mapper[x] if x in mapper else mapper[self.unk])
            if self.add_cls:
                tmp_seq.append(mapper[self.cls])
            res.append(tmp_seq)
        return res


def collate_fn(data, device):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            padded_seqs[i, : lengths[i]] = seq
        return padded_seqs, lengths

    data_by_key = {key: [d[key] for d in data] for key in data[0].keys()}

    src_utt, _ = merge(data_by_key["utterance"])
    y_slots, y_lengths = merge(data_by_key["slots"])
    intent = torch.LongTensor(data_by_key["intent"])

    return {
        "utterances": src_utt.to(device),
        "intents": intent.to(device),
        "y_slots": y_slots.to(device),
        "slots_len": torch.LongTensor(y_lengths).to(device),
    }


def make_datasets(dataset_dir="dataset/ATIS", dev_size=0.10, seed=42):
    train_raw, dev_raw, test_raw = load_dataset(
        dataset_dir=dataset_dir,
        dev_size=dev_size,
        seed=seed,
    )
    lang = build_lang(train_raw, dev_raw, test_raw)

    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    return train_dataset, dev_dataset, test_dataset, lang, train_raw, dev_raw, test_raw


def make_dataloaders(
    device,
    dataset_dir="dataset/ATIS",
    train_batch_size=128,
    eval_batch_size=64,
    seed=42,
    data_seed=None,
    dev_size=0.10,
):
    # Keep the train/dev split fixed across repeated runs while allowing the model
    # initialization and the training shuffle order to change with `seed`.
    if data_seed is None:
        data_seed = seed

    train_dataset, dev_dataset, test_dataset, lang, train_raw, dev_raw, test_raw = make_datasets(
        dataset_dir=dataset_dir,
        dev_size=dev_size,
        seed=data_seed,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    collate = partial(collate_fn, device=device)

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

    return train_loader, dev_loader, test_loader, lang, train_raw, dev_raw, test_raw
