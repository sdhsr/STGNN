import torch
import torch.nn as nn
import torch.optim as optim

# 假设有一个简单的线性模型
model = nn.Linear(10, 1)  # 输入特征维度为10，输出维度为1

# 损失函数
loss_fn = nn.MSELoss()  # 均方误差损失

# 优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 假设有一些输入数据和真实标签
inputs = torch.randn(5, 10)  # 5个样本，每个样本有10个特征
targets = torch.randn(5, 1)  # 5个样本的目标值

# 前向传播
outputs = model(inputs)

# 计算损失
loss = loss_fn(outputs, targets)

# 反向传播
loss.backward()

# 参数更新
optimizer.step()

# 清除梯度（在每次迭代结束时进行，因为默认梯度会累加）
optimizer.zero_grad()