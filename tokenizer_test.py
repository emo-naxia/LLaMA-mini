from tokenizers.implementations import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer("trained_tokenizer-vocab.txt")

text = "你好，大语言模型的训练是怎样的？"
enc = tokenizer.encode(text)

print("Input text:", text)
print("Token IDs:", enc.ids)
print("Tokens   :", enc.tokens)
