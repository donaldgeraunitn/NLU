from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


# Extend the original GPT-2 attention layer with trainable LoRA adapters
# for the query, key, and value projections. The pretrained projections
# remain in place and their outputs receive low-rank additive updates.
class CustomGPT2Attention(GPT2Attention):
    def __init__(self, config, rank, alpha, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)

        self.rank = rank
        self.alpha = alpha
        # Scale each low-rank update according to the LoRA formulation.
        self.scaling = alpha / rank

        # Each adapter factorizes a full projection update into two small
        # matrices: A reduces the dimension and B projects it back.
        self.lora_q_A = nn.Linear(self.embed_dim, rank, bias=False)
        self.lora_q_B = nn.Linear(rank, self.embed_dim, bias=False)

        self.lora_k_A = nn.Linear(self.embed_dim, rank, bias=False)
        self.lora_k_B = nn.Linear(rank, self.embed_dim, bias=False)

        self.lora_v_A = nn.Linear(self.embed_dim, rank, bias=False)
        self.lora_v_B = nn.Linear(rank, self.embed_dim, bias=False)

        # Initializing B to zero makes every LoRA update initially zero,
        # preserving the pretrained model output at the start of tuning.
        nn.init.normal_(self.lora_q_A.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lora_k_A.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lora_v_A.weight, mean=0.0, std=0.02)

        nn.init.zeros_(self.lora_q_B.weight)
        nn.init.zeros_(self.lora_k_B.weight)
        nn.init.zeros_(self.lora_v_B.weight)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        # Preserve GPT-2 attention behavior and add the LoRA contribution
        # immediately after the standard query, key, and value projections.
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask

            query = query + self.lora_q_B(self.lora_q_A(hidden_states)) * self.scaling
            key = key + self.lora_k_B(self.lora_k_A(encoder_hidden_states)) * self.scaling
            value = value + self.lora_v_B(self.lora_v_A(encoder_hidden_states)) * self.scaling
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

            query = query + self.lora_q_B(self.lora_q_A(hidden_states)) * self.scaling
            key = key + self.lora_k_B(self.lora_k_A(hidden_states)) * self.scaling
            value = value + self.lora_v_B(self.lora_v_A(hidden_states)) * self.scaling

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Keep the standard GPT-2 cache path for compatibility, although
        # caching is disabled during fine-tuning in functions.py.
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Start from Hugging Face GPT2LMHeadModel and replace every self-attention
# module with the LoRA-enabled version while copying pretrained weights.
class GPT2_LoRA(GPT2LMHeadModel):
    def __init__(self, *model_args, rank, alpha, **model_kwargs):
        super().__init__(*model_args, **model_kwargs)

        self.rank = rank
        self.alpha = alpha

        # strict=False loads the original attention weights while allowing
        # the newly introduced LoRA matrices to keep their initialization.
        for block in self.transformer.h:
            custom_attn = CustomGPT2Attention( self.config, rank=rank, alpha=alpha, layer_idx=getattr(block.attn, "layer_idx", None) )

            custom_attn.load_state_dict(block.attn.state_dict(), strict=False)
            block.attn = custom_attn

        # Freeze the pretrained GPT-2 parameters: only LoRA matrices are optimized.
        for name, param in self.named_parameters():
            param.requires_grad = "lora_" in name

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


# Report the parameter reduction obtained by training only LoRA adapters.
def param_stats(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"total params: {total:,}")
    print(f"trainable params: {trainable:,}")
    print(f"frozen params: {total - trainable:,}")
    return total, trainable
