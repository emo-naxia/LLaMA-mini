from tokenizers import Tokenizer as RawTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from pathlib import Path

from pathlib import Path

class Tokenizer:
    def __init__(self, model_path, bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]"):
        vocab_path = Path("data") / "trained_tokenizer" / "vocab.txt"

        assert vocab_path.exists(), f"❌ 词表文件不存在: {vocab_path}"

        # 其余保持不变...


        # 加载词表 vocab
        vocab = {}
        with open(vocab_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                token = line.strip()
                vocab[token] = i

        # 初始化 tokenizer
        self.tokenizer = RawTokenizer(WordLevel(vocab, unk_token=unk_token))
        self.tokenizer.pre_tokenizer = Whitespace()

        # 设置特殊 token 模板处理方式
        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{bos_token} $A {eos_token}",
            pair=f"{bos_token} $A {eos_token} $B:1 {eos_token}:1",
            special_tokens=[
                (bos_token, vocab[bos_token]),
                (eos_token, vocab[eos_token]),
            ]
        )

        # 保存特殊 token 的 ID
        self.bos_id = vocab[bos_token]
        self.eos_id = vocab[eos_token]
        self.unk_id = vocab[unk_token]

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)
