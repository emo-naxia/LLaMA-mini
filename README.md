# LLaMA-mini
一个轻量级的 LLaMA 模型，从零实现 tokenizer、模型、训练与推理，支持中英文，Mac 上可运行
LLaMA-mini/
├── .git/                         # Git 配置文件夹（自动生成）
├── .gitattributes                # Git LFS 配置文件
├── .DS_Store                     # macOS 系统文件，可忽略
├── README.md                     # 项目说明文档
├── config.py                     # 统一配置路径（模型、tokenizer）
├── requirements.txt              # 项目依赖
│
├── checkpoints/                  # 中间模型保存目录
│   └── model_epoch1.pt           # ✅ 中间权重文件（大文件，建议忽略上传）
│
├── data/                         # 数据文件夹
│   ├── train_data.pt             # ✅ 编码后的训练数据（大文件）
│   ├── tokenizers_corpus.txt     # ✅ 原始训练语料
│   └── trained_tokenizer/        # ✅ tokenizer 保存目录
│       ├── tokenizer.json
│       └── vocab.txt
│
├── models/                       # 模型结构模块
│   ├── __init__.py
│   ├── attention.py
│   ├── decoder_block.py          # ✅ 解码器模块
│   ├── mlp.py
│   ├── model.py
│   ├── model_config.py
│   └── rmsnorm.py                # ✅ RMSNorm 层定义
│
├── tokenizer.py                  # 自定义 tokenizer 类
├── tokenizer_test.py             # 测试 tokenizer 效果
├── train_tokenizer.py            # 训练 tokenizer
├── prepare_dataset.py            # 编码语料为 tensor 数据
│
├── train.py                      # 训练主程序
├── save_model.py                 # 保存模型权重
├── generate.py                   # 推理文本生成
├── model_epoch23.pt              # ✅ 最终训练模型（大文件）
