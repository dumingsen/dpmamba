import os
import uuid
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.utils.data
from print import print
from printt import printt
#========================
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(embeddings, labels, epoch):
    plt.clf()
    perplexity = min(30, len(embeddings) // 3)

    # 设置 learning_rate 为 200.0
    tsne = TSNE(n_components=2, perplexity=perplexity, init='random', random_state=42, learning_rate=200.0)
    
    node_pos = tsne.fit_transform(embeddings.detach().cpu().numpy())

    # 获取类别数量
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # 根据类别数量选择合适的颜色映射
    cmap = plt.get_cmap('coolwarm', num_classes)

    scatter = plt.scatter(node_pos[:, 0], node_pos[:, 1], c=labels, cmap=cmap, s=5)
    plt.title(f"t-SNE Embeddings at Epoch {epoch}")
    
    plt.colorbar(scatter, ticks=unique_labels)  # 添加颜色条，显示类别

    # 保存图像到文件
    save_dir = '/root/SimTSC-main/SimTSC-main/tsne'
    os.makedirs(save_dir, exist_ok=True)  # 确保文件夹存在
    plt.savefig(os.path.join(save_dir, f'tsne_embeddings_epoch_{epoch}.png'))
    plt.close()  # 关闭图形，释放内存

def visualize_embeddingso(embeddings, labels, epoch):
    plt.clf()
    perplexity = min(30, len(embeddings) // 3)
    
    # 设置 learning_rate 为 200.0（与当前默认值一致）
    tsne = TSNE(n_components=2, perplexity=perplexity, init='random', random_state=42, learning_rate=200.0)
    #, perplexity=perplexity, learning_rate='auto', random_state=42
    
    node_pos = tsne.fit_transform(embeddings.detach().cpu().numpy())
    
    scatter=plt.scatter(node_pos[:, 0], node_pos[:, 1], c=labels, cmap='coolwarm', s=5)
    plt.title(f"t-SNE Embeddings at Epoch {epoch}")

    plt.colorbar(scatter)

    # 保存图像到文件
    save_dir='/root/SimTSC-main/SimTSC-main/tsne'
    plt.savefig(os.path.join(save_dir, f'tsne_embeddings_epoch_{epoch}.png'))
    plt.close()  # 关闭图形，释放内存
    plt.show()

def visualize(embeddings, labels, epoch):
    plt.clf()
    tsne = TSNE(n_components=2)
    node_pos = tsne.fit_transform(embeddings)

    scatter = plt.scatter(node_pos[:, 0], node_pos[:, 1], c=labels, cmap='coolwarm', s=100)
    plt.title(f'Epoch {epoch}')
    plt.colorbar(scatter)
    plt.pause(0.5)  # 暂停一段时间，用于动态展示

#===================
class SimTSCTrainer:
    def __init__(self, device, logger):
        self.device = device
        self.logger = logger
        self.tmp_dir = 'tmp'
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def fit(self, model, X, y, train_idx, distances, K, alpha, test_idx=None, report_test=True, batch_size=4, epochs=50):#batch128
        self.K = K
        self.alpha = alpha
        print('===================fit')
        print('textid', test_idx.shape)
        print('train_idx, distances, K, alpha', len(train_idx), distances.shape, K, alpha)
        
        train_batch_size = min(batch_size // 2, len(train_idx))
        other_idx = np.array([i for i in range(len(X)) if i not in train_idx])
        print('other_idx', len(other_idx))

        other_batch_size = min(batch_size - train_batch_size, len(other_idx))
        train_dataset = Dataset(train_idx)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=1)
        
        if report_test:
            print('======================report test')
            test_batch_size = min(batch_size // 2, len(test_idx))
            other_idx_test = np.array([i for i in range(len(X)) if i not in test_idx])
            other_batch_size_test = min(batch_size - test_batch_size, len(other_idx_test))
            test_dataset = Dataset(test_idx)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=1)
        
        self.adj = torch.from_numpy(distances.astype(np.float32))
        self.X, self.y = torch.from_numpy(X), torch.from_numpy(y)
        file_path = os.path.join(self.tmp_dir, str(uuid.uuid4()))

        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=4e-3)
        best_acc = 0.0

        # 创建保存所有节点嵌入的矩阵
        all_embeddings = torch.zeros((len(X), 15)).to(self.device)

        # Training
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            for sampled_train_idx in train_loader:
                print('====================================train', sampled_train_idx)
                sampled_other_idx = np.random.choice(other_idx, other_batch_size, replace=False)
                idx = np.concatenate((sampled_train_idx, sampled_other_idx))
                print('idx', len(idx))

                _X = self.X[idx].to(self.device)
                _y = self.y[sampled_train_idx.long()].to(self.device)
                _adj = self.adj[idx][:, idx]

                print('_X, _y, _adj', _X.shape, _y.shape, _adj.shape)
                outputs = model(_X, _adj, K, alpha)
                
                loss = F.nll_loss(outputs[:len(sampled_train_idx)], _y)
                loss.backward()
                optimizer.step()

                # # 提取当前批次的节点嵌入，并保存
                # with torch.no_grad():
                #     out = model.get_node_embeddings(_X, _adj)
                #     all_embeddings[idx] = out.detach()  # 保存每个批次的嵌入

                # 直接保存模型最后一层的输出作为嵌入
                all_embeddings[idx] = outputs.detach()
            
            model.eval()
            print('all embed', all_embeddings.shape)
            # 可视化嵌入
            #visualize_embeddings(all_embeddings, self.y, epoch)

            
            acc = compute_accuracy(model, self.X, self.y, self.adj, self.K, self.alpha, train_loader, self.device, other_idx, other_batch_size)
            if acc >= best_acc:
                best_acc = acc
                torch.save(model.state_dict(), file_path)
            if report_test:
                test_acc = compute_accuracy(model, self.X, self.y, self.adj, self.K, self.alpha, test_loader, self.device, other_idx_test, other_batch_size_test)
                self.logger.log('--> Epoch {}: loss {:5.4f}; accuracy: {:5.4f}; best accuracy: {:5.4f}; test accuracy: {:5.4f}'.format(epoch, loss.item(), acc, best_acc, test_acc))
            else:
                self.logger.log('--> Epoch {}: loss {:5.4f}; accuracy: {:5.4f}; best accuracy: {:5.4f}'.format(epoch, loss.item(), acc, best_acc))

        model.load_state_dict(torch.load(file_path))
        model.eval()
        os.remove(file_path)

        # 返回所有节点的嵌入表示
        return  model#all_embeddings,

    
    def test(self, model, test_idx, batch_size=128):
        test_batch_size = min(batch_size//2, len(test_idx))
        other_idx_test = np.array([i for i in range(len(self.X)) if i not in test_idx])
        other_batch_size_test = min(batch_size - test_batch_size, len(other_idx_test))
        test_dataset = Dataset(test_idx)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=1)
        acc = compute_accuracy(model, self.X, self.y, self.adj, self.K, self.alpha, test_loader, self.device, other_idx_test, other_batch_size_test)
        return acc.item()

def compute_accuracy(model, X, y, adj, K, alpha, loader, device, other_idx, other_batch_size):
    #                                     test_loader, self.device, other_idx_test, other_batch_size_test
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx in loader:
            sampled_other_idx = np.random.choice(other_idx, other_batch_size, replace=False)
            idx = np.concatenate((batch_idx, sampled_other_idx))
            _X, _y, _adj = X[idx].to(device), y[idx][:len(batch_idx)].to(device), adj[idx][:,idx]

            outputs = model(_X, _adj, K, alpha)
            preds = outputs[:len(batch_idx)].max(1)[1].type_as(_y)
            _correct = preds.eq(_y).double()
            correct += _correct.sum()
            total += len(batch_idx)
    acc = correct / total
    return acc
#===============================================================================================
from mamba.experiments.exp_basic import Exp_Basic
from mamba.model.S_Mamba import Model
class choose_model(Exp_Basic):
    def __init__(self, args):
        super(choose_model, self).__init__(args)

    def _build_model(self):
        model_mamba = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model_mamba = nn.DataParallel(model_mamba, device_ids=self.args.device_ids)
        return model_mamba
    # def ma(self):
    #     outputs = self.model_mamba(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
from node_classification.models import GFASTKAN_Nodes
from optuna.trial import Trial
def graphtrans(adj): 
    
    adj = adj
    # 示例带权邻接矩阵
    # adj = torch.tensor([[0.0, 0.5, 0.0],
    #                     [0.5, 0.0, 1.0],
    #                     [0.0, 1.0, 0.0]], dtype=torch.float)
    # 初始化边列表和边权重
    edge_index = []
    edge_weights = []

    # 遍历邻接矩阵
    num_nodes = adj.size(0)
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = adj[i][j]
            if weight > 0:  # 只考虑权重大于 0 的边
                edge_index.append((i, j))
                edge_weights.append(weight.item())  # 将权重转换为 Python 标量

    # 转换为张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).T  # 转置为 (2, E)
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    # print("Edge Index:")
    # print(edge_index)
    # print("\nEdge Weights:")
    # print(edge_weights)
    return edge_index, edge_weights
class SimTSC(nn.Module):
    def __init__(self, input_size, nb_classes, args1, num_layers=1, n_feature_maps=128, dropout=0.5):
        super(SimTSC, self).__init__()
        self.num_layers = num_layers

        # self.block_1 = ResNetBlock(input_size, n_feature_maps)
        # self.block_2 = ResNetBlock(n_feature_maps, n_feature_maps)
        # self.block_3 = ResNetBlock(n_feature_maps, n_feature_maps)

        if self.num_layers == 1:
            self.gc1 = GraphConvolution(n_feature_maps, nb_classes)
        elif self.num_layers == 2:
            self.gc1 = GraphConvolution(n_feature_maps, n_feature_maps)
            self.gc2 = GraphConvolution(n_feature_maps, nb_classes)
            self.dropout = dropout
        elif self.num_layers == 3:
            self.gc1 = GraphConvolution(n_feature_maps, n_feature_maps)
            self.gc2 = GraphConvolution(n_feature_maps, n_feature_maps)
            self.gc3 = GraphConvolution(n_feature_maps, nb_classes)
            self.dropout = dropout
        self.args1=args1
        self.mamba=Model(self.args1).float()

        #trial=Trial
        hidden_layers = 1 #trial.suggest_int('hidden_layers', 1, 4)
        hidden_channels  = 128 #512#trial.suggest_categorical('hidden_channels', [16, 32, 64, 128])
        skip = True
        grid_size = 3 #trial.suggest_int('grid_size', 3, 5)
        num_classes=nb_classes
        num_features=128 #512 
        conv_type='gin'
        self.kangin = GFASTKAN_Nodes( conv_type = conv_type,  num_layers = hidden_layers, num_features = num_features, hidden_channels= hidden_channels,
                        num_classes = num_classes, skip = skip, grid_size=grid_size)
        self.linear1 = nn.Linear(1, 128)#
    def forward(self, x, adj, K, alpha):
        print('===============================model',)
        print('x0',x.shape)#[128, 3, 205])
        ranks = torch.argsort(adj.cpu(), dim=1)#原本无cpu
        sparse_index = [[], []]
        sparse_value = []
        for i in range(len(adj)):
            _sparse_value = []
            for j in ranks[i][:K]:
                sparse_index[0].append(i)
                sparse_index[1].append(j)
                _sparse_value.append(1/np.exp(alpha*adj[i][j].cpu().numpy()))###
            _sparse_value = np.array(_sparse_value)
            _sparse_value /= _sparse_value.sum()
            sparse_value.extend(_sparse_value.tolist())
        sparse_index = torch.LongTensor(sparse_index)
        sparse_value = torch.FloatTensor(sparse_value)
        adj = torch.sparse.FloatTensor(sparse_index, sparse_value, adj.size())
        device = self.gc1.bias.device
        adj = adj.to(device)

        x = x.permute(0, 2, 1)
        print('x1', x.shape)#([128, 205, 3])
        x = self.mamba(x)#x2 torch.Size([3, 512]) self.mamba(x)[0]
        print('x2',x.shape)#x2 torch.Size([128, 3, 512]) [4, 400, 128]
        # x = self.block_1(x)
        # x = self.block_2(x)
        # x = self.block_3(x)
        x = x.permute(0, 2, 1)
        print('x3',x.shape)#4, 128, 400])
        x = F.avg_pool1d(x, x.shape[-1]).squeeze()
        print('x4',x.shape)#x torch.Size([128, 512]) [4, 128])
        #input()

        edge_index, edge_weight=graphtrans(adj)
        edge_index=edge_index.to(device)
        edge_weight=edge_weight.unsqueeze(1).to(device)
        edge_weight = self.linear1(edge_weight)
        if self.num_layers == 1:
            x = self.kangin(x, edge_index, edge_weight)#.to(device)
            #x = self.gc1(x, adj)
        elif self.num_layers == 2:
            x = F.relu(self.gc1(x, adj))
            print('x5',x.shape)#5 torch.Size([4, 128])
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj)
            print('x6',x.shape)# torch.Size([4, 4])
        elif self.num_layers == 3:
            x = F.relu(self.gc1(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(self.gc2(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc3(x, adj)

        print('gin x', x.shape) # [4, 4]) batch class
        x = F.log_softmax(x, dim=1)
        print('x5',x.shape)#x5 torch.Size([128, 20]) [4, 4])
        #input()
        return x

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.expand = True if in_channels < out_channels else False

        self.conv_x = nn.Conv1d(in_channels, out_channels, 7, padding=3)
        self.bn_x = nn.BatchNorm1d(out_channels)
        self.conv_y = nn.Conv1d(out_channels, out_channels, 5, padding=2)
        self.bn_y = nn.BatchNorm1d(out_channels)
        self.conv_z = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn_z = nn.BatchNorm1d(out_channels)

        if self.expand:
            self.shortcut_y = nn.Conv1d(in_channels, out_channels, 1)
        self.bn_shortcut_y = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        B, _, L = x.shape
        out = F.relu(self.bn_x(self.conv_x(x)))
        out = F.relu(self.bn_y(self.conv_y(out)))
        out = self.bn_z(self.conv_z(out))

        if self.expand:
            x = self.shortcut_y(x)
        x = self.bn_shortcut_y(x)
        out += x
        out = F.relu(out)
       
        return out

class Dataset(torch.utils.data.Dataset):
    def __init__(self, idx):
        self.idx = idx

    def __getitem__(self, index):
        return self.idx[index]

    def __len__(self):
        return len(self.idx)

