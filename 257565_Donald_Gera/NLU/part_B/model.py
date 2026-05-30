import torch
import torch.nn as nn
from transformers import AutoModel


class GPT2(nn.Module):
    def __init__(self, model_name, slots_size, n_intents, pad_token_id, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.config.pad_token_id = pad_token_id
        self.encoder.config.use_cache = False

        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(hidden_size, slots_size)
        self.intent_out = nn.Linear(hidden_size, n_intents)

    def forward(self, input_ids, attention_mask):
        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        slots = self.slot_out(self.dropout(hidden_states))

        last_token_positions = attention_mask.sum(dim=1) - 1
        batch_indexes = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_tokens = hidden_states[batch_indexes, last_token_positions]
        intents = self.intent_out(self.dropout(last_tokens))

        return slots, intents


class BERT(nn.Module):
    def __init__(self, model_name, slots_size, n_intents, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(hidden_size, slots_size)
        self.intent_out = nn.Linear(hidden_size, n_intents)

    def forward(self, input_ids, attention_mask):
        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        slots = self.slot_out(self.dropout(hidden_states))
        intents = self.intent_out(self.dropout(hidden_states[:, 0]))

        return slots, intents


def make_model(model_type, model_name, slots_size, n_intents, tokenizer, dropout=0.1):
    if model_type == "gpt2":
        return GPT2(
            model_name=model_name,
            slots_size=slots_size,
            n_intents=n_intents,
            pad_token_id=tokenizer.pad_token_id,
            dropout=dropout,
        )
    if model_type == "bert":
        return BERT(
            model_name=model_name,
            slots_size=slots_size,
            n_intents=n_intents,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported model type: {model_type}")
