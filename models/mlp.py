import torch
import torch.nn as nn
from models.model_config import ModelConfig

import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 自动推导 hidden_dim，如果没有手动指定的话
        hidden_dim = (
            config.hidden_dim
            if config.hidden_dim is not None
            else int(2 * config.dim * 4 / 3)  # 官方 LLaMA2 的默认值：约为 dim * 2.66
        )
        # 对齐为 multiple_of 的整数倍（利于加速）
        hidden_dim = int((hidden_dim + config.multiple_of - 1) // config.multiple_of * config.multiple_of)

        self.gate_proj = nn.Linear(config.dim, hidden_dim, bias=False)
        self.up_proj   = nn.Linear(config.dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.dim, bias=False)

    def forward(self, x):
        # LLaMA 的 MLP 激活函数是 SiLU(x) * x （带 gating 的 GLU 结构）
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# 测试
if __name__ == "__main__":
    args = ModelConfig()
    mlp = MLP(args)
    x = torch.randn(1, 10, args.dim)
    out = mlp(x)
    print("MLP output shape:", out.shape)
