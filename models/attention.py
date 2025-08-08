import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_config import ModelConfig

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    作用：构造每个位置和每个维度的旋转角频率（复数形式）
    预先计算好所有位置的位置编码因子（旋转角度）
    返回的是：形状为 [max_seq_len, dim // 2] 的复数
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # [seq_len, dim // 2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 转成复数：cos + i*sin
    return freqs_cis  # shape: [max_seq_len, dim // 2]

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    """
    对 Q / K 应用旋转位置编码（RoPE）
    x: [batch, seq_len, n_heads, head_dim]
    freqs_cis: [seq_len, head_dim // 2]
    """
    # 将 freqs_cis 移到和 x 一样的设备上（CPU/MPS/GPU）
    freqs_cis = freqs_cis.to(x.device)
    # 转为复数：最后一维变为 (real, imag)
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # [B, T, H, D/2]
    # 修剪 freqs_cis 长度，并扩展成可广播形状
    freqs_cis = freqs_cis[:x_.size(1)]  # [T, D//2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # [1, T, 1, D//2]
    # 应用旋转
    x_out = x_ * freqs_cis  # [B, T, H, D/2] * [1, T, 1, D/2]
    x_out = torch.view_as_real(x_out).flatten(-2)  # -> [B, T, H, D]
    return x_out.type_as(x)

class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads

        # Q, K, V 和输出 projection
        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)

        # 预先计算 RoPE 的旋转因子
        self.freqs_cis = precompute_freqs_cis(self.head_dim, config.max_seq_len)

    def forward(self, x):
        B, T, C = x.size()  # B=batch_size, T=seq_len, C=dim

        # 得到 Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        # 应用 RoPE
        q = apply_rotary_emb(q, self.freqs_cis)
        k = apply_rotary_emb(k, self.freqs_cis)

        # 转置用于后续 matmul（注意顺序）
        q = q.transpose(1, 2)  # [B, n_heads, T, head_dim]
        k = k.transpose(1, 2)  # [B, n_kv_heads, T, head_dim]
        v = v.transpose(1, 2)  # [B, n_kv_heads, T, head_dim]

        # 如果启用 GQA（n_heads ≠ n_kv_heads），复制 KV
        if self.n_kv_heads != self.n_heads:
            repeat_factor = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)  # 在 head 维度上重复
            v = v.repeat_interleave(repeat_factor, dim=1)

        # 缩放点积注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, n_heads, T, T]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [B, n_heads, T, head_dim]

        # 合并所有 head，输出维度 [B, T, C]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn_output)
if __name__ == "__main__":
    args = ModelConfig()
    attn = Attention(args)
    x = torch.randn(1, 10, args.dim)  # 模拟一个输入：1个样本，10个token
    out = attn(x)
    print("Attention output shape:", out.shape)
