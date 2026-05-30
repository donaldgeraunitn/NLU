import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        # As in the LoRA paper: A starts randomly and B starts from zero.
        # Therefore, the initial LoRA update BA is exactly zero.
        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(self.A(x)) * self.scaling


class LoRAQKV(nn.Module):
    def __init__(self, c_attn, hidden_size, rank, alpha):
        super().__init__()

        # Hugging Face GPT-2 stores Q, K and V in one combined c_attn layer.
        # Its frozen output has size 3 * hidden_size and is split afterwards.
        self.c_attn = c_attn
        self.lora_q = LoRA(hidden_size, hidden_size, rank, alpha)
        self.lora_k = LoRA(hidden_size, hidden_size, rank, alpha)
        self.lora_v = LoRA(hidden_size, hidden_size, rank, alpha)

    def forward(self, x):
        frozen_qkv = self.c_attn(x)

        lora_q = self.lora_q(x)
        lora_k = self.lora_k(x)
        lora_v = self.lora_v(x)
        lora_qkv = torch.cat([lora_q, lora_k, lora_v], dim=-1)

        return frozen_qkv + lora_qkv


class GPT2_LoRA(nn.Module):
    def __init__(self, model_name="openai-community/gpt2", rank=4, alpha=32):
        super().__init__()

        self.model_name = model_name
        self.rank = rank
        self.alpha = alpha

        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.config.use_cache = False

        # Freeze every parameter loaded from the pre-trained GPT-2 model.
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        # Add fresh trainable LoRA adapters to every self-attention block.
        hidden_size = self.model.config.n_embd
        for block in self.model.transformer.h:
            block.attn.c_attn = LoRAQKV(c_attn=block.attn.c_attn, hidden_size=hidden_size, rank=rank, alpha=alpha)

        self._check_trainable_parameters()

    def _check_trainable_parameters(self):
        unexpected = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and ".lora_" not in name
        ]

        if unexpected:
            raise RuntimeError(
                "Only LoRA parameters should be trainable, but found: "
                + ", ".join(unexpected)
            )

    def forward(self, input_ids, labels=None):
        return self.model(input_ids=input_ids, labels=labels)


def get_trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def lora_state_dict(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
        if ".lora_" in name
    }


def load_lora_state_dict(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if unexpected_keys:
        raise ValueError(f"Unexpected LoRA checkpoint keys: {unexpected_keys}")

    # Missing keys are expected: the checkpoint intentionally stores only the
    # small adapter tensors and reloads the frozen GPT-2 weights from Hugging Face.
    return missing_keys
