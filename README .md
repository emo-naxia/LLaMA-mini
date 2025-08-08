# LLaMA-mini

> 🦙 A lightweight LLaMA-like model built from scratch: from Tokenizer to Architecture, Training, and Inference. Fully runnable on macOS with MPS acceleration.

> 🧠 一个轻量级 LLaMA 模型，从零开始实现分词器、模型结构、训练流程与推理代码，支持在 macOS 上运行。

---

## 🚀 1. Project Overview | 项目简介

LLaMA-mini is a minimal and educational version of a LLaMA-like large language model, intended for students, researchers, or developers who want to understand and build LLMs step by step.

LLaMA-mini 是一个教学型的 LLaMA 轻量实现项目，适合希望深入理解大语言模型结构与训练原理的学习者与开发者。

---

## 🧱 2. Features | 项目特点

- ✅ Fully Custom Tokenizer（自定义中文分词器）
- ✅ Rotary Attention with FlashAttention toggle（支持旋转位置编码与 FlashAttention）
- ✅ LLaMA-style MLP & LayerNorm
- ✅ Compatible with MPS / CPU / CUDA（适配 Mac / Linux / Windows）
- ✅ Full Training & Inference Pipeline（包含完整训练与推理流程）

---

## 📦 3. Installation | 安装依赖

```bash
conda create -n LLM_LLaMA2 python=3.10
conda activate LLM_LLaMA2
pip install -r requirements.txt
```

确保你已经安装了 PyTorch，并启用了 MPS / CUDA / CPU。

---

## 📂 4. Project Structure | 项目结构

```bash
LLaMA-mini/
├── data/                  # 存放训练语料和生成的dataset
├── models/                # 模型结构模块（注意力、MLP、配置）
│   ├── model.py
│   ├── model_config.py
│   ├── attention.py
│   ├── mlp.py
│   └── __init__.py
├── tokenizer.py          # 自定义Tokenizer类
├── train_tokenizer.py    # 分词器训练脚本
├── tokenizer_test.py     # 测试分词器效果
├── prepare_dataset.py    # 将文本语料编码为训练数据
├── train.py              # 主训练脚本
├── save_model.py         # 保存训练模型权重
├── generate.py           # 文本生成推理脚本
├── requirements.txt      # 项目依赖
└── README.md             # 项目说明文档
```

---

## 📝 5. Prepare Your Dataset | 数据准备

将你的语料放入 `data/corpus.txt` 文件中，每行为一条语句。

```bash
python train_tokenizer.py       # 训练 tokenizer
python tokenizer_test.py        # 检查分词效果
python prepare_dataset.py       # 编码语料为训练数据
```

输出将生成：

- `train_data.pt`：用于训练的 tensor 数据
- `vocab.txt`：分词器词表

---

## 🎯 6. Train the Model | 模型训练

```bash
python train.py
```

你可以在 `train.py` 中修改训练轮数、批大小、学习率等超参数。

训练完成后，模型将保存为：

```bash
model_epoch1.pt
```

---

## 🤖 7. Run Inference | 推理生成

```bash
python generate.py
```

根据输入提示生成文本：

```
💬 请输入一句开头：你好
🧠 模型生成结果：你好，欢迎使用 LLaMA-mini 模型...
```

---

## 📌 8. Notes | 说明

- 训练轮数少 / 语料量小时，生成结果可能较随机。
- 可进一步训练更多 epoch、扩展语料、调优超参数。

---

## 📮 9. License | 许可证

本项目仅用于学习与研究，遵循 MIT 开源协议。

---

欢迎 Star、Fork 与二次开发！🚀

> Made with ❤️ by emo

南京工业大学计算科学与技术专业本科生