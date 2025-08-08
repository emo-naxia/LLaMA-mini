from tokenizers.implementations import BertWordPieceTokenizer
import os
# 语料路径
corpus_path = "data/tokenizer_corpus.txt"
assert os.path.exists(corpus_path), "语料文件不存在！"

# 初始化 tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=True,  # ✅ 中文适配关键
    strip_accents=False,
    lowercase=False
)

# 开始训练（建议至少10000行文本）
tokenizer.train(
    files=[corpus_path],
    vocab_size=6144,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
)

# 保存
tokenizer.save_model(".", "trained_tokenizer")

print("✅ WordPiece 中文 tokenizer 训练完成！")
tokenizer.save("tokenizer.json")
