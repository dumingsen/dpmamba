import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 定义一个简单的图神经网络模型
class GNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        # 图卷积层（简单版，忽略归一化）
        h = torch.matmul(adj, x)
        h = self.relu(self.fc1(h))
        h = torch.matmul(adj, h)
        h = self.fc2(h)
        return h

# 可视化节点及其连接
def visualize_with_edges(node_embeddings, adj_matrix, labels):
    plt.figure()
    
    # 获取节点的二维嵌入坐标
    x_coords = node_embeddings[:, 0].detach().numpy()
    y_coords = node_embeddings[:, 1].detach().numpy()
    
    # 绘制节点
    for i in range(len(x_coords)):
        if labels[i] == 0:
            plt.scatter(x_coords[i], y_coords[i], color='green')  # 类别0的节点
        else:
            plt.scatter(x_coords[i], y_coords[i], color='blue')   # 类别1的节点
    
    # 绘制连接线，根据邻接矩阵
    num_nodes = adj_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i, j] > 0:  # 如果节点之间有连接
                plt.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], color='black', linewidth=0.5)
    
    # 突出显示一些特殊的节点，比如红色
    plt.scatter(x_coords[0], y_coords[0], color='red')  # 假设0号节点为特殊节点
    plt.scatter(x_coords[-1], y_coords[-1], color='red')  # 假设最后一个节点为特殊节点
    
    plt.show()

# 训练和可视化函数
def train_and_visualize(G, num_epochs=20, batch_size=10):
    # 将NetworkX的图转为邻接矩阵
    adj = nx.to_numpy_matrix(G)
    adj = torch.tensor(adj, dtype=torch.float32)

    # 随机生成节点特征（假设每个节点有3维特征）
    num_nodes = adj.shape[0]
    node_features = torch.randn(num_nodes, 3)

    # 假设节点的真实标签是二分类的
    labels = torch.randint(0, 2, (num_nodes,))

    # 初始化模型、损失函数和优化器
    model = GNN(in_features=3, hidden_features=5, out_features=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 模拟训练过程
    for epoch in range(num_epochs):
        model.train()

        # 正向传播
        out = model(node_features, adj)
        
        # 计算损失
        loss = criterion(out, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # 获取模型输出的节点嵌入
    model.eval()
    with torch.no_grad():
        node_embeddings = model.fc1(node_features)  # 获取隐藏层的节点嵌入（用于可视化）

    # 可视化节点及其连接
    visualize_with_edges(node_embeddings, adj, labels)

# 示例：创建一个随机图并训练模型
G = nx.erdos_renyi_graph(50, 0.1)  # 生成一个随机图
train_and_visualize(G, num_epochs=20, batch_size=10)
