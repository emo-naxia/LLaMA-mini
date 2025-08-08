import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from models.model_config import ModelConfig

from rmsnorm import RMSNorm
from attention import Attention
from mlp import MLP

class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attn = Attention(config)

        self.mlp_norm = RMSNorm(config.dim, config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x):
        # Attention 层 + 残差
        h = self.attn_norm(x)
        x = x + self.attn(h)

        # MLP 层 + 残差
        h = self.mlp_norm(x)
        x = x + self.mlp(h)

        return x


# 测试
if __name__ == "__main__":
    args = ModelConfig()
    block = DecoderBlock(args)
    x = torch.randn(1, 10, args.dim)
    out = block(x)
    print("DecoderBlock output shape:", out.shape)
