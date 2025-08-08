# LLaMA-mini 使用指南（从 0 开始）

> ⚙️ 本文档为复现与使用 LLaMA-mini 项目的完整操作指南，适用于从未运行过该项目的用户。

---

## 🧰 1. 环境准备

### ✅ 安装 Anaconda（推荐）
前往官网下载安装：
https://www.anaconda.com/download

### ✅ 创建 Python 虚拟环境（建议 Python 3.10）

```bash
conda create -n LLM_LLaMA2 python=3.10
conda activate LLM_LLaMA2
```

### ✅ 安装依赖

项目根目录下执行：

```bash
pip install -r requirements.txt
```

确保安装了 PyTorch，并支持你的计算设备（CPU / CUDA / MPS）：
https://pytorch.org/get-started/locally/

---

## 📁 2. 项目结构（已更新）

```bash
LLaMA-mini/
├── checkpoints/                  # 保存中间训练权重（.pt）
├── data/                         # 存放训练语料、数据集
│   ├── tokenizers_corpus.txt     # 自定义语料文件
│   └── trained_tokenizer/        # 分词器保存目录
│       ├── vocab.txt             # 词表
│       └── tokenizer.json        # 分词器结构
├── models/                       # 模型结构文件
│   ├── attention.py
│   ├── decoder_block.py
│   ├── model.py
│   ├── model_config.py
│   ├── mlp.py
│   ├── rmsnorm.py
│   └── __init__.py
├── config.py                     # 全局配置文件（路径等）
├── generate.py                   # 推理脚本
├── prepare_dataset.py            # 将语料编码为训练数据
├── requirements.txt              # Python依赖
├── save_model.py                 # 模型保存与加载
├── tokenizer.py                  # 分词器实现
├── tokenizer_test.py             # 分词器测试脚本
├── train.py                      # 主训练脚本
├── train_tokenizer.py            # 分词器训练脚本
└── README.md                     # 项目说明文档
```

---

## 📦 3. 分词器准备

### ✅ 修改或替换训练语料

将语料写入：

```
data/tokenizers_corpus.txt
```

一行一句，建议越干净越好。

### ✅ 训练分词器

```bash
python train_tokenizer.py
```

### ✅ 测试分词器

```bash
python tokenizer_test.py
```

### ✅ 生成训练数据

```bash
python prepare_dataset.py
```

会生成 `data/train_data.pt`

---

## 🧠 4. 模型训练

默认训练参数在 `train.py` 中：

```python
BATCH_SIZE = 8
SEQ_LEN = 128
EPOCHS = 23
LR = 2e-4
```

可以根据设备能力、语料大小进行修改。

### ✅ 启动训练

```bash
python train.py
```

训练权重将保存至：

```bash
model_epoch23.pt
```

---

## 🤖 5. 推理文本生成

修改 `config.py` 中的模型路径（默认已设为 `model_epoch23.pt`），然后运行：

```bash
python generate.py
```

输入提示词，如：

```
💬 请输入一句开头：你好
🧠 模型生成结果：你好，欢迎使用 LLaMA-mini 模型...
```

---

## ⚙️ 6. 常见问题

- ❗ 如果报错 `vocab.txt 不存在`：请确保你已经训练并保存过 tokenizer。
- ❗ 如果生成结果全是 `[UNK]`：语料可能太少或模型未充分训练。
- ❗ `.pt` 文件过大无法上传 GitHub：请使用 `.gitignore` 排除。

---

## 🌟 7. 注意事项与建议

- 建议用干净、简短的语料训练 tokenizer，效果更好。
- 模型训练时务必保持设备稳定、避免中断。
- 可尝试增加训练轮数或调大模型尺寸以获得更好效果。

---

## 🧾 8. 授权 License

本项目仅用于学习与研究，遵循 MIT 开源协议。

---

> 📍 项目地址：https://github.com/emo-naxia/LLaMA-mini
>  
> ✨ Author: emo（南京工业大学）