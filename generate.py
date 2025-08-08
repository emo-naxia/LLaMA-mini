import torch
from models.model import LLaMAModel
from models.model_config import ModelConfig
from tokenizer import Tokenizer
from config import MODEL_PATH, VOCAB_PATH, TOKENIZER_PATH

# ==== 自动适配设备 ====
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# ==== 初始化 tokenizer ====
tokenizer = Tokenizer(model_path=TOKENIZER_PATH, bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]")

# ==== 初始化模型 ====
config = ModelConfig()
model = LLaMAModel(config)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.to(DEVICE)  # ✅ 将模型移动到设备
model.eval()

# ==== 简单推理函数 ====
def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(DEVICE)  # ✅ 放入设备
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_id:
                break
    return tokenizer.decode(input_ids[0].tolist())

# ==== 交互 ====
prompt = input("💬 请输入一句开头：")
result = generate_text(prompt)
print("🧠 模型生成结果：", result)