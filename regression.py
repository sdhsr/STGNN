import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 使用sklearn生成一个二分类数据集
X, y = make_classification(
    n_samples=1000,  # 样本数量
    n_features=2,    # 特征数量（二维数据）
    n_classes=2,     # 类别数量（二分类）
    n_informative=2, # 有效特征数目（影响分类的特征数量）
    n_redundant=0,   # 冗余特征数目（没有冗余特征）
    random_state=42  # 随机种子，确保每次运行数据集相同
)

# 将数据转换为PyTorch的tensor格式，方便在PyTorch中进行计算
X = torch.tensor(X, dtype=torch.float32)  # 特征数据
y = torch.tensor(y, dtype=torch.float32)  # 标签数据


# 使用train_test_split将数据集划分为训练集和测试集，测试集占20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 定义一个简单的逻辑回归模型（继承自nn.Module）
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        # 逻辑回归是一个线性模型，这里定义一个全连接层（线性变换）
        self.linear = nn.Linear(input_dim, 1)  # 输入特征数，输出1个预测值

    def forward(self, x):
        # 对线性层输出进行sigmoid激活，确保输出是0到1之间的概率
        return torch.sigmoid(self.linear(x))  # 使用sigmoid函数将输出转换为概率


input_dim = X.shape[1]  # 输入数据的特征数量，X的列数（二维数据）
model = LogisticRegressionModel(input_dim)  # 初始化模型


# 使用BCELoss作为损失函数，适用于二分类问题（Binary Cross-Entropy Loss）
criterion = nn.BCELoss()

# 使用随机梯度下降（SGD）优化器来优化模型的参数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 学习率设为0.01

# 设置训练的轮数
num_epochs = 5000

# 开始训练模型
for epoch in range(num_epochs):
    # 前向传播：计算模型的输出
    outputs = model(X_train).squeeze()  # 调用模型得到预测值，并去掉多余的维度
    loss = criterion(outputs, y_train)  # 计算损失

    # 反向传播：计算梯度并更新模型参数
    optimizer.zero_grad()  # 清除之前的梯度
    loss.backward()  # 计算当前梯度
    optimizer.step()  # 更新模型参数

    # 每训练10个epoch打印一次损失值
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 在测试时不需要计算梯度（节省内存和计算资源）
with torch.no_grad():
    # 预测测试集的输出
    y_pred_prob = model(X_test).squeeze()  # 获得概率值
    y_pred = (y_pred_prob >= 0.5).float()  # 将概率大于等于0.5的预测为1，小于0.5的预测为0
    # 计算准确率
    accuracy = accuracy_score(y_test.numpy(), y_pred.numpy())
    print(f'Accuracy on test data: {accuracy:.4f}')


def plot_decision_boundary(X, y, model):
    # 计算数据范围并生成网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # 将网格数据转化为tensor并预测每个点的类别
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        probs = model(grid).reshape(xx.shape).numpy()  # 得到网格上每个点的预测概率

    # 绘制决策边界
    plt.contourf(xx, yy, probs, alpha=0.8, cmap='coolwarm')  # 绘制决策边界
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm')  # 绘制数据点
    plt.title('Decision Boundary')  # 图标题
    plt.show()

# 调用函数绘制决策边界
plot_decision_boundary(X.numpy(), y.numpy(), model)
