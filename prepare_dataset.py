from tokenizers.implementations import BertWordPieceTokenizer
import torch
import os

# 路径配置
tokenizer_path = "trained_tokenizer-vocab.txt"
corpus_path = "data/tokenizer_corpus.txt"
output_path = "data/train_data.pt"

# 加载 tokenizer
tokenizer = BertWordPieceTokenizer(tokenizer_path)

# 读取语料
with open(corpus_path, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

input_ids_list = []
for line in lines:
    enc = tokenizer.encode(line)
    ids = enc.ids
    input_ids_list.append(torch.tensor(ids, dtype=torch.long))

# 拼接为一个大长序列
all_ids = torch.cat(input_ids_list)

# 保存
os.makedirs("data", exist_ok=True)
torch.save(all_ids, output_path)

print(f"✅ 数据处理完成，保存至 {output_path}，共 {len(all_ids)} tokens")
