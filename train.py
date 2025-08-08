
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from models.model import LLaMAModel
from models.model_config import ModelConfig
from config import MODEL_PATH  # ✅ 引入模型保存路径

# ==== 超参数 ====
BATCH_SIZE = 8
SEQ_LEN = 128
EPOCHS = 23
LR = 2e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# 设置种子，确保可复现
torch.manual_seed(42)

print(f"✅ Using device: {DEVICE}")

# ==== 加载数据 ====
class LMDataset(Dataset):
    def __init__(self, data_path, seq_len):
        self.data = torch.load(data_path)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return x, y

dataset = LMDataset("data/train_data.pt", SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==== 初始化模型 ====
config = ModelConfig()
model = LLaMAModel(config).to(DEVICE)

# ==== 优化器 & 损失函数 ====
optimizer = optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# ==== 训练循环 ====
import time
start_time = time.time()

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for step, (x, y) in enumerate(dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)  # [B, T, V]
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))  # Flatten for CE Loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 10 == 0:
            print(f"[Epoch {epoch + 1}] Step {step}/{len(dataloader)} Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"✅ Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

# ==== 保存最终模型 ====
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ 最终模型已保存为: {MODEL_PATH}")

print(f"⏱️ 总训练耗时: {time.time() - start_time:.2f} 秒")
