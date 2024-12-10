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

def generate_pass_file(filename='D:\\sdh\\passes.txt'):
    with open(filename, 'w') as file:
        for i in range(1, 10000):
            formatted_number = f"PASS{i:04d}"
            file.write(formatted_number + '\n')


if __name__ == "__main__":
    generate_pass_file()




