import torch
import torch.nn as nn
import math
from einops import einsum, rearrange, reduce
from typing import Optional

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std = torch.sqrt(torch.tensor(2 / (in_features + out_features)))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)
        
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor: 
        output = einsum(self.weight, x, "d_out d_in, batch sequence d_in -> batch sequence d_out")
        return output
    
class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(
        self,
        token_ids: torch.tensor
    ) -> torch.Tensor:
        output = self.weight[token_ids]
        return output
    
class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: Optional[float] = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.device] = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
    
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.square(x).mean(dim=-1, keepdim=True) + self.eps)
        output = x / rms * self.weight
        return output.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        if d_ff is None:
            # round up to multiple of 64
            d_ff = int((8/3 * d_model + 63) // 64 * 64)
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        temp = self.w1.forward(x)
        output = temp * torch.sigmoid(temp)
        output *= self.w3.forward(x)    
        output = self.w2.forward(output)
        return output

class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        d = theta ** (2 * torch.arange(0, d_k // 2, device=device) / d_k).unsqueeze(0)
        s = torch.arange(max_seq_len, device=device).unsqueeze(-1)
        t = s / d
        sin_vals = torch.sin(t)
        cos_vals = torch.cos(t)
        self.register_buffer("sin_vals", sin_vals)
        self.register_buffer("cos_vals", cos_vals)
        
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor
    ) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        sin_vals = self.sin_vals[token_positions]
        cos_vals = self.cos_vals[token_positions]
        x_even = x1 * cos_vals - x2 * sin_vals
        x_odd = x1 * sin_vals + x2 * cos_vals
        
        output = torch.stack([x_even, x_odd], dim=-1).flatten(start_dim=-2)
        return output
    
def Softmax(
    x: torch.Tensor,
    i: int
) -> torch.Tensor:
    x -= x.max(dim=i, keepdim=True).values
    x = x.exp()
    return x / x.sum(dim=i, keepdim=True)

def Attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor]
) -> torch.Tensor:
    d_k = q.shape[-1]
    qk = einsum(q, k, "... n d_k, ... m d_k -> ... n m")
    qk /= math.sqrt(d_k)
    if mask is not None:
        qk = qk.masked_fill(mask == False, float("-inf"))
    wei = Softmax(qk, -1)
    output = einsum(wei, v, "... n m, ... m d_v -> ... n d_v")
    return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: Optional[RoPE] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads 
        self.q_proj = Linear(d_model, d_model, device=device)
        self.k_proj = Linear(d_model, d_model, device=device)
        self.v_proj = Linear(d_model, d_model, device=device)
        self.output_proj = Linear(d_model, d_model, device=device)
        self.rope = rope
    def forward(
        self,
        x: torch.Tensor,
        token_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seq_len = x.shape[-2]
        
        q = rearrange(self.q_proj.forward(x), "... sequence (h d_k) -> ... h sequence d_k", d_k=self.head_dim, h=self.num_heads)
        k = rearrange(self.k_proj.forward(x), "... sequence (h d_k) -> ... h sequence d_k", d_k=self.head_dim, h=self.num_heads)
        v = rearrange(self.v_proj.forward(x), "... sequence (h d_v) -> ... h sequence d_v", d_v=self.head_dim, h=self.num_heads)
        

        # wq = rearrange(self.q_proj, "... (num_heads d_k) d_in -> ... num_heads d_k d_in", d_k = self.head_dim, num_heads = self.num_heads)
        # wk = rearrange(self.k_proj, "... (num_heads d_k) d_in -> ... num_heads d_k d_in", d_k = self.head_dim, num_heads = self.num_heads)
        # wv = rearrange(self.v_proj, "... (num_heads d_v) d_in -> ... num_heads d_v d_in", d_v = self.head_dim, num_heads = self.num_heads)
        # # turn this into 1 matmul?
        # q = einsum(wq, x, "... num_heads d_k d_in, ... sequence d_in -> ... num_heads sequence d_k")
        # k = einsum(wk, x, "... num_heads d_k d_in, ... sequence d_in -> ... num_heads sequence d_k")
        # v = einsum(wv, x, "... num_heads d_v d_in, ... sequence d_in -> ... num_heads sequence d_v")
        if token_positions is not None:
            q = self.rope.forward(q, token_positions)
            k = self.rope.forward(k, token_positions)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        output = rearrange(Attention(q, k, v, mask), "... num_heads sequence d_v -> ... sequence (num_heads d_v)")
        
        # output = einsum(self.output_proj, output, "... d_model hd_v, ... sequence hd_v -> ... sequence d_model")
        output = self.output_proj.forward(output)

        return output
    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RoPE,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope, device=device)
        self.ffn = SwiGLU(d_model, d_ff, device=device)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        seq_len = x.shape[-2]
        x = x + self.attn.forward(self.ln1(x), torch.arange(0, seq_len).unsqueeze(0))
        x = x + self.ffn.forward(self.ln2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: int,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device)
        rope = RoPE(rope_theta, d_model // num_heads, context_length, device=device)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, rope, device=device) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device=device)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = x[..., :self.context_length, :]
        
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer.forward(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits