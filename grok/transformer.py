import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, List, Tuple
import math  # Add missing import for math functions

class Linear(nn.Linear):
    def __init__(self, *args, weight_noise: float = 0.0, bias=True, **kwargs):
        super().__init__(*args, bias=bias, **kwargs)
        self.weight_noise = weight_noise

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight_noise > 0 and self.training:
            weight = self.weight + torch.randn_like(self.weight) * self.weight_noise
        else:
            weight = self.weight
        return F.linear(input, weight, self.bias)


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, elementwise_affine=True, **kwargs):
        super().__init__(*args, elementwise_affine=elementwise_affine, **kwargs)


class Embedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx=None, **kwargs):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, **kwargs)


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_key: int, weight_noise: float = 0.0) -> None:
        super().__init__()
        self.d_key = d_key

        self.Wq = Linear(d_model, d_key, bias=False, weight_noise=weight_noise)
        self.Wk = Linear(d_model, d_key, bias=False, weight_noise=weight_noise)
        self.Wv = Linear(d_model, d_key, bias=False, weight_noise=weight_noise)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
        save_activations: bool = False,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:

        attn = torch.matmul(self.Wq(queries), self.Wk(keys).transpose(-2, -1)) / math.sqrt(self.d_key)  # Use math.sqrt instead of **0.5

        if mask is not None:
            attn.masked_fill_(mask == 0, float("-inf"))

        attn = self.softmax(attn)

        result = torch.matmul(attn, self.Wv(values))  # shape = (max_context_len, d_key)

        return result, None, None  # Updated to remove unused return values


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, weight_noise: float = 0.0) -> None:
        super().__init__()
        d_key = d_model // heads

        self.attn_heads = nn.ModuleList([AttentionHead(d_model, d_key, weight_noise) for _ in range(heads)])
        self.Wo = Linear(d_model, d_model, bias=False, weight_noise=weight_noise)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor = None,
        save_activations=False,
    ) -> Tuple[torch.Tensor, None, None]:

        head_results = [head(queries, keys, values, mask) for head in self.attn_heads]
        multihead_result = torch.cat([output[0] for output in head_results], dim=-1)

        return self.Wo(multihead_result), None, None  # Updated to remove unused return values


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        multiplier: int = 4,
        activation: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()
         
        d_ff = d_model * multiplier

        self.ffn = nn.Sequential(
            Linear(d_model, d_ff, bias=True, weight_noise=weight_noise),
            getattr(nn, activation)(),
            Linear(d_ff, d_model, bias=True, weight_noise=weight_noise),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        dropout: float = 0.1,
        activation: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, heads, weight_noise=weight_noise)
        self.self_attn_norm = LayerNorm(d_model)
        self.ffn = FFN(d_model, activation=activation, weight_noise=weight_noise)
        self.ffn_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        save_activations: bool = False,
    ) -> Tuple[torch.Tensor, None, None]:
        a1, _, _ = self.self_attn(x, x, x, self_attn_mask)  # Unpack the outputs of MultiHeadAttention
        a1 = self.dropout(a1)
        a1 = self.self_attn_norm(x + a1)

        a2 = self.ffn(a1)
        a2 = self.dropout(a2)
        a2 = self.ffn_norm(a1 + a2)

        return a2, None, None  # Updated to remove unused return values


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        num_blocks: int,
        dropout: float = 0.1,
        activation: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model, heads, dropout, activation=activation, weight_noise=weight_noise
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        save_activations=False,
    ) -> Tuple[torch.Tensor, None, None]:

        a = x
        for block in self.blocks:
            a, _, _ = block(a, self_attn_mask, save_activations=save_activations)  # Unpack the outputs of DecoderBlock
        return a, None, None  # Updated to remove unused return values


class Transformer(nn.Module):
    def __init__(
        self,
        n_layers: int = 4,
        n_heads: int = 4,
        d_model: int = 256,
        dropout: float = 0.1,
        max_context_len: int = 1024,
        vocab_len: int = 2000,
        activation: str = "relu",
        weight_noise: float = 0.0,
    ) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout
        self.max_context_len = max_context_len
        self.activation = activation

        self.vocab_len = vocab_len

        self.embedding = Embedding(vocab_len, d_model)
        self.register_buffer(
            "position_encoding", self._position_encoding(max_context_len, d_model)
        )
        self.register_buffer("self_attn_mask", self.make_mask(max_context_len))

        self.decoder = Decoder(
            d_model,
            n_heads,
            n_layers,
            dropout,
            activation=activation,
            weight_noise=weight_noise,
        )

        self.linear = Linear(d_model, vocab_len, bias=False, weight_noise=weight_noise)

    @staticmethod
    def make_mask(context_len: int) -> torch.Tensor:
        return torch.ones([context_len, context_len]).tril()

    @classmethod
    def _position_encoding(cls, context_len: int, d_model: int) -> torch.Tensor:
        rows = [
            torch.tensor(  # Use torch.tensor instead of tensor
                [
                    math.sin(pos / (10000 ** (i / d_model)))
                    if i % 2 == 0
                    else math.cos(pos / (10000 ** ((i - 1) / d_model)))
                    for i in range(d_model)
                ]
            )
            for pos in range(context_len)
        ]
        stack = torch.stack(rows, dim=1)

        return stack.T  # type: ignore

    def embed(self, indices: torch.Tensor) -> torch.Tensor:
        context_len = indices.shape[-1]
        pe = self.position_encoding[:context_len, :]  # type: ignore

        embedded = self.embedding(indices)

        return pe + embedded

    def forward(
        self,
        x: torch.Tensor,
        pos: int = None,
        save_activations: bool = False,
    ) -> torch.Tensor:
        """parameters:
        x:  (rank-1 tensor) vocab indices of decoder input token
                     sequence"""

        # Make sure sampling inputs are on the correct device
        x = x.to(self.embedding.weight.device)

        # make_attention mask
        this_max_context_len = x.shape[-1]
        self_attn_mask = self.self_attn_mask[  # type: ignore
            :this_max_context_len, :this_max_context_len
        ]

        # Decode
        x = self.embed(x)
        decoded, _, _ = self.decoder(x, self_attn_mask, save_activations=save_activations)

        # Return predictions for specific token
        if pos is not None:
            decoded = decoded[:, pos, :]

        y_hat = self.linear(decoded)
        return y_hat
        
