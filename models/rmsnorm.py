import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import torch#数值计算
import torch.nn as nn#神经网络模块（我们用它来写 RMSNorm）
from models.model_config import ModelConfig

class RMSNorm(nn.Module):#定义了一个新的 PyTorch 层，叫做 RMSNorm，继承自 nn.Module
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习参数

    def _norm(self, x):#RMSNorm的核心公式
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# 测试模块是否正常
if __name__ == "__main__":
    from model_config import ModelConfig#从 model_config.py 导入参数配置
    args = ModelConfig()

    norm = RMSNorm(args.dim, args.norm_eps)
    x = torch.randn(1, 50, args.dim)
    output = norm(x)
    print("Output shape:", output.shape)
