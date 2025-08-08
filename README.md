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
├── checkpoints/                   # 模型检查点（未上传至 GitHub）
│   └── model_epoch1.pt            # epoch1 训练权重
├── data/                          # 语料和训练数据
│   ├── tokenizers_corpus.txt      # 分词器训练语料
│   ├── train_data.pt              # 编码后的训练数据（未上传至 GitHub）
│   └── trained_tokenizer/         # 保存分词器
│       ├── vocab.txt              # 分词器词表
│       └── tokenizer.json         # 分词器模型
├── models/                        # 模型结构模块
│   ├── attention.py
│   ├── decoder_block.py
│   ├── model.py
│   ├── model_config.py
│   ├── mlp.py
│   ├── rmsnorm.py
│   └── __init__.py
├── config.py                      # 配置模型路径和常量
├── generate.py                    # 推理脚本
├── prepare_dataset.py            # 编码训练数据
├── requirements.txt              # 项目依赖
├── save_model.py                 # 保存权重
├── tokenizer.py                  # 自定义 Tokenizer 类
├── tokenizer_test.py             # 测试分词器效果
├── train.py                      # 模型训练脚本
└── train_tokenizer.py            # 分词器训练脚本
```

---

## 📝 5. Prepare Your Dataset | 数据准备

将语料文件放入 `data/tokenizers_corpus.txt` 中，每行一句。

```bash
python train_tokenizer.py       # 训练 tokenizer
python tokenizer_test.py        # 检查分词效果
python prepare_dataset.py       # 编码语料为训练数据
```

---

## 🎯 6. Train the Model | 模型训练

```bash
python train.py
```

训练好的模型将保存在 `checkpoints/` 目录中，如：

```
checkpoints/model_epoch1.pt
```

---

## 🤖 7. Run Inference | 推理生成

```bash
python generate.py
```

你可以输入一句开头文本，模型将自动补全后续：

```
💬 请输入一句开头：你好
🧠 模型生成结果：你好，欢迎使用 LLaMA-mini 模型...
```

---

## 📌 8. Notes | 说明

- 如果模型效果不好，可以继续增加训练轮数、扩展语料或调节参数。
- 模型和训练数据未上传至 GitHub（因文件过大），请自行训练或加载本地文件。

---

## 📮 9. License | 许可证

本项目仅用于学习与研究，遵循 MIT 开源协议。

---

欢迎 Star、Fork 与改进本项目！🚀

> Made with ❤️ by emo

南京工业大学 计算科学与技术专业 本科生
