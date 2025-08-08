from transformers import PretrainedConfig

class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"

    def __init__(
        self,
        dim: int = 768,#模型维度;768（即每个 token 会被编码成一个 768维向量）
        n_layers: int = 12,#Transformer 的层数
        n_heads: int = 16,#注意力头数
        n_kv_heads: int = 8,#用于 Grouped-Query Attention 的 KV头数量
        vocab_size: int = 6144,#词表大小（token 的种类数）
        hidden_dim: int = None,#FFN的中间隐藏层维度（MLP）;默认为 None（后面自动推算）
        multiple_of: int = 64,#控制 hidden_dim 是否为 64 的倍数;提高硬件效率
        norm_eps: float = 1e-5,#用于 RMSNorm 的防除0项;默认 1e-5
        max_seq_len: int = 512,#最大序列长度;512 个 token
        dropout: float = 0.0,#dropout 概率;训练时防止过拟合
        flash_attn: bool = True,#是否启用FlashAttention
        **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)

# 测试是否成功
if __name__ == "__main__":
    args = ModelConfig()
    print(args)
