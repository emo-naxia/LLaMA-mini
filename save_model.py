
import torch
from models.model import LLaMAModel
from models.model_config import ModelConfig
from config import MODEL_PATH

# 初始化模型结构
config = ModelConfig()
model = LLaMAModel(config)

# 加载已有权重（可替换路径）
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

# 保存权重（这里是覆盖，也可以改文件名）
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ 模型已保存为: {MODEL_PATH}")
