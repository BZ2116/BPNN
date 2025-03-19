"""
author:Bruce Zhao
data:2024
"""
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# 1. 加载替代数据集（加利福尼亚房价）
data = fetch_california_housing()
X, y = data.data, data.target.reshape(-1, 1)  # 特征形状: (20640, 8), 标签形状: (20640, 1)
#原始形状为 (20640,)，是一维数组，reshape(-1, 1)将标签数据从一维数组转换为二维数组，转换后的形状为 (20640, 1)。

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 标准化（注意：仅用训练集拟合scaler）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# 5. 定义模型（注意输入维度调整为8）
class BPNN(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, output_size=1):#output_size=1：输出层维度为1，适用于回归任务（预测房价）
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 6. 训练参数
model = BPNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 600

# 7. 训练循环
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 8. 测试
model.eval()
with torch.no_grad():
    test_pred = model(X_test)
    test_loss = criterion(test_pred, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

# 9. 预测新数据示例
X_new = [[ 0.98, -0.57, 0.43, -0.26, 0.31, -0.12, 0.05, -0.03]]  # 假设新样本（形状需为8维）
X_new_scaled = scaler.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)
prediction = model(X_new_tensor)
print(f"Predicted Price: {prediction.item():.2f} (单位：10万美元)")