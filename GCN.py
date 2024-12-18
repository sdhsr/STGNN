# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# # 图卷积层定义
# class GraphConvolution(nn.Module):
#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
#         if bias:
#             self.bias = nn.Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         # 初始化权重
#         nn.init.xavier_uniform_(self.weight)
#         if self.bias is not None:
#             nn.init.zeros_(self.bias)
#
#     def forward(self, input, adj):
#         # 计算图卷积：AXW + b
#         support = torch.mm(input, self.weight)
#         output = torch.mm(adj, support)
#         if self.bias is not None:
#             output += self.bias
#         return output
#
#
# # 定义GCN模型
# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()
#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout
#
#     def forward(self, x, adj):
#         # 第一层图卷积 + 激活 + Dropout
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         # 第二层图卷积 (输出层)
#         x = self.gc2(x, adj)
#         return F.log_softmax(x, dim=1)
#
#
# # 示例：使用一个简单的图
# if __name__ == "__main__":
#     # 节点特征矩阵 (4个节点，每个节点3维特征)
#     features = torch.tensor([[1, 0, 2],
#                               [0, 1, 0],
#                               [1, 1, 1],
#                               [0, 0, 1]], dtype=torch.float)
#
#     # 邻接矩阵 (无自环，未归一化)
#     adj = torch.tensor([[0, 1, 1, 0],
#                         [1, 0, 1, 1],
#                         [1, 1, 0, 1],
#                         [0, 1, 1, 0]], dtype=torch.float)
#
#     # 加入自环，并归一化邻接矩阵
#     adj = adj + torch.eye(adj.size(0))
#     degree = adj.sum(dim=1)
#     adj = adj / degree.unsqueeze(1)
#
#     # 模型定义
#     model = GCN(nfeat=3, nhid=4, nclass=2, dropout=0.5)
#
#     # 示例标签 (4个节点分别属于2类中的一类)
#     labels = torch.tensor([0, 1, 0, 1])
#
#     # 优化器
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#
#     # 训练
#     for epoch in range(50):
#         model.train()
#         optimizer.zero_grad()
#         output = model(features, adj)
#         loss = F.nll_loss(output, labels)
#         loss.backward()
#         optimizer.step()
#
#         print(f"Epoch {epoch+1}, Loss: {loss.item()}")
#
#     # 测试
#     model.eval()
#     _, pred = model(features, adj).max(dim=1)
#     print(f"Predicted Labels: {pred.tolist()}")


import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv


def visualize_graph(G, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, node_color=color, cmap="Set2")
    plt.savefig("D:\\Java\\USTGCN\\graph.png")


# 1.图的结构
# Karate Clubs数据集包含一个无向图，有34个节点和78条边
# 每个节点代表俱乐部中的一个成员
# 边表示两个成员之间的友谊关系
# 2.节点属性
# 节点具有一个类别标签，0,1,2,3，表示属于哪一类
# 3.任务
# 根据成员之间的关系决定成员的类别
dataset = KarateClub()
print(f"Dataset: {dataset}:")
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]
# 输出Data(x=[34, 34], edge_index=[2, 156], y=[34], train_mask=[34])
# x：形状为 [34, 34] 的特征矩阵，表示有34个节点，每个节点有34个特征
# edge_index：形状为 [2, 156]的邻接矩阵，前面说到图中的78条边是无向边，而PyTorch Geometric的 edge_index默认将无向边视为两条方向相反的有向边
#             start→end两个序列，所以有两行
# y：形状为 [34] 的标签向量，表示每个节点的标签
# train_mask：形状为 [34] 的布尔向量，表示哪些节点用于训练
print(data)

# to_undirected=True无向图
G = to_networkx(data, to_undirected=True)
# 输出一个networkx图
visualize_graph(G, data.y)


class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        # 图卷积中变化的只有节点的特征维度，邻接矩阵永远不变；dataset.num_features=34
        # in_channels节点输入特征维度，out_channels节点输出特征维度
        self.conv1 = GCNConv(in_channels=dataset.num_features, out_channels=4)
        self.conv2 = GCNConv(in_channels=4, out_channels=4)
        self.conv3 = GCNConv(in_channels=4, out_channels=2)
        self.classifier = nn.Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        y1 = self.conv1(x, edge_index)
        y1 = torch.relu(y1)
        y2 = self.conv2(y1, edge_index)
        y2 = torch.relu(y2)
        y3 = self.conv3(y2, edge_index)
        y3 = torch.relu(y3)

        out = self.classifier(y3)

        return out, y3


if __name__ == '__main__':

    model = GCN()
    # 定义损失函数
    loss = nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 200
    for epoch in range(epochs):
        # 清空优化器的梯度
        optimizer.zero_grad()
        # 设置为训练模式
        model.train()
        # 前向传播
        out, h = model(data.x, data.edge_index)
        # 计算损失
        # 只看mask为True的节点，因此是半监督
        loss_train = loss(out[data.train_mask], data.y[data.train_mask])
        # 反向传播
        loss_train.backward()
        # 更新模型参数
        optimizer.step()
        print(loss_train)








