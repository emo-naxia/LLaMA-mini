import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from models.model_config import ModelConfig

from decoder_block import DecoderBlock
from rmsnorm import RMSNorm


class LLaMAModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 词嵌入层（token embedding）
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        # 堆叠多个 DecoderBlock
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])

        # 最后一层 RMSNorm（官方设计）
        self.norm = RMSNorm(config.dim, config.norm_eps)

        # 语言建模头（输出每个 token 的 logits）
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

    def compute_loss(self, input_ids, labels):
        """
        用于训练时计算 cross entropy loss
        input_ids: [B, T]
        labels: [B, T]
        """
        logits = self.forward(input_ids)  # [B, T, vocab_size]
        vocab_size = logits.size(-1)

        # 将 logits 和 labels reshape 成 [B*T, vocab_size] 和 [B*T]
        loss = nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100  # 可选：忽略 padding
        )
        return loss

    def forward(self, input_ids):
        # input_ids: [batch, seq_len]
        x = self.tok_embeddings(input_ids)  # [B, T, dim]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)  # [B, T, dim]
        logits = self.output(x)  # [B, T, vocab_size]

        return logits


# ✅ 测试代码
if __name__ == "__main__":
    args = ModelConfig()
    model = LLaMAModel(args)

    # 构造一个假输入：1 个样本，长度为 10，每个 token 是 vocab 内的 ID（随机）
    input_ids = torch.randint(0, args.vocab_size, (1, 10))

    output = model(input_ids)
    print("Output logits shape:", output.shape)
