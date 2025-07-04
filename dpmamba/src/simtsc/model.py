import datetime
import os
import uuid
import math
from datetime import datetime
import numpy as np

from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.utils.data
from print import print
from printt import printt
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

def compute_classes(model, X, y, adj, K, alpha, loader, device, other_idx, other_batch_size):
        all_outputs = []
        all_preds = []
        
        with torch.no_grad():
            for batch_idx in loader:
                sampled_other_idx = np.random.choice(other_idx, other_batch_size, replace=False)
                idx = np.concatenate((batch_idx, sampled_other_idx))
                _X, _y, _adj = X[idx].to(device), y[idx][:len(batch_idx)].to(device), adj[idx][:, idx]

                outputs = model(_X, _adj, K, alpha)

                outputs=outputs[:len(batch_idx)]
                preds = outputs[:len(batch_idx)].max(1)[1].type_as(_y)
                print('outputs preds', outputs.shape, preds.shape)

                # 累积输出和预测
                all_outputs.append(outputs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

        # 将输出和预测合并为一个数组
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)

        return all_outputs, all_preds

class SimTSCTrainer:
    def __init__(self, device, logger,args):
        self.device = device
        self.logger = logger
        self.tmp_dir = 'tmp'
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)


        # 确保在类中定义这些变量
        self.loss_history = []  # 用于保存每个 epoch 的损失
        self.acc_history = []   # 用于保存每个 epoch 的准确率

        self.args=args

        self.args.dataset=args.dataset

        self.accuracies = []
        self.losses = []
    

    def visualize_node_classification_tsne(self, model, epoch):
        dataset_name = self.args.dataset

        # 将模型设置为评估模式
        model.eval()

        # 获取特征和邻接矩阵
        features = self.X.cpu().numpy()
        adj_matrix = self.adj.cpu().numpy()
        # 构建图
        # 根据 test_idx 提取对应的子矩阵
        new_adj_matrix = adj_matrix[np.ix_(self.test_idx, self.test_idx)]

        #=========== 计算节点的输出
        # 获取测试集的预测类别
        test_batch_size = min(self.batch_size // 2, len(self.test_idx))
        other_idx_test = np.array([i for i in range(len(self.X)) if i not in self.test_idx])
        other_batch_size_test = min(self.batch_size - test_batch_size, len(other_idx_test))
        
        test_dataset = Dataset(self.test_idx)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=1)
        
        outputs, predicted_classes = compute_classes (model, self.X, self.y, self.adj, self.K, self.alpha, test_loader, 
                                                   self.device, other_idx_test, other_batch_size_test)

        printt('outputs, predicted_classes', outputs.shape, predicted_classes.shape)
        #outputs, predicted_classes torch.Size([7, 15]) torch.Size([3])

        # ======================获取节点的预测类别
        #_, predicted_classes = torch.max(outputs, dim=1)

        # 使用 t-SNE 降维
        n_samples = outputs.shape[0]
        perplexity = min(30, n_samples - 1)  # 确保 perplexity 小于样本数量

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings = tsne.fit_transform(outputs)#(47, 2)
        printt('embeddings', embeddings.shape)

        # 获取每个节点的标签
        labels = self.y[self.test_idx].cpu().numpy()
        printt('标签', len(labels))
        num_classes_real = len(np.unique(labels))  # 获取类别数量
        printt('实际标签种类数量', num_classes_real)
        
        # 确保 predicted_classes 是整数并与 embeddings 的长度一致
        predicted_classes = predicted_classes#.cpu().numpy()
        
        # 检查 predicted_classes 的长度和类型
        if len(predicted_classes) != len(embeddings):
            raise ValueError("Length of predicted_classes must equal the number of embeddings.")

   

       # 假设 embeddings 是 t-SNE 降维后的结果，predicted_classes 是类别标签
        num_classes = len(np.unique(predicted_classes))  # 获取类别数量
        printt('预测标签种类数量', num_classes)

        plt.figure(figsize=(12, 8))

        # 使用离散颜色映射
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels,##predicted_classes
                            cmap=plt.cm.get_cmap('tab10', num_classes),  # 使用离散颜色图
                            alpha=0.8, s=100)  # 增加透明度和散点大小

        # 添加颜色条
        cbar = plt.colorbar(scatter)
        cbar.set_label('Predicted Class')

        # # 添加标签（仅在 embeddings 范围内）
        # for i in range(len(embeddings)):
        #     if i < len(labels):  # 确保不会超出 labels 的索引范围
        #         plt.annotate(str(labels[i]), (embeddings[i, 0], embeddings[i, 1]), fontsize=8)

        # 添加标签（可选）
        # for i in range(len(labels)):
        #     plt.annotate(str(labels[i]), (embeddings[i, 0], embeddings[i, 1]), fontsize=8)

        for i in range(len(predicted_classes)):
            plt.annotate(str(predicted_classes[i]), (embeddings[i, 0], embeddings[i, 1]), fontsize=8)

        # 连接每个节点到其最近的三个邻居
        # n_neighbors=1
        # for idx in range(len(self.test_idx)):
        #     # 获取当前节点的邻接列表，并找出最近的三个邻居
        #     neighbors = np.argsort(new_adj_matrix[idx])[-(n_neighbors + 1):-1]  # 获取最近的三个邻居（忽略自身）
        #     for neighbor in neighbors:
        #         # 连接 idx 和 neighbor
        #         plt.plot([embeddings[idx, 0], embeddings[neighbor, 0]], 
        #                 [embeddings[idx, 1], embeddings[neighbor, 1]], 'k-', alpha=0.5)

        plt.title(f'Node Classification with t-SNE at Epoch {epoch}')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.axis('off')

        # 保存可视化图
        save_path = f'/root/SimTSC-main/SimTSC-main/visual_tsne/{dataset_name}/'
        os.makedirs(save_path, exist_ok=True)  # 自动创建目录
        plt.savefig(os.path.join(save_path, f'tsne_node_classification_epoch_{epoch}.png'))
        plt.show()
        printt('tsne 保存')
        plt.close()


    def visualize_node_clustering(self, model, X, y, distances,epoch):
        dataset_name = self.args.dataset
        self.test_labels = self.y[self.test_idx].cpu().numpy()
        self.class_test=np.unique(self.test_labels)
        self.num_class_test=len(self.class_test)


        # 将模型设置为评估模式
        model.eval()

        # 获取特征和邻接矩阵
        features = self.X.cpu().numpy()
        adj_matrix = self.adj.cpu().numpy()
        # 构建图
        # 根据 test_idx 提取对应的子矩阵
        new_adj_matrix = adj_matrix[np.ix_(self.test_idx, self.test_idx)]

        #=========== 计算节点的输出
        # 获取测试集的预测类别
        test_batch_size = min(self.batch_size // 2, len(self.test_idx))
        other_idx_test = np.array([i for i in range(len(self.X)) if i not in self.test_idx])
        other_batch_size_test = min(self.batch_size - test_batch_size, len(other_idx_test))
        
        test_dataset = Dataset(self.test_idx)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=1)
        
        outputs, predicted_classes = compute_classes (model, self.X, self.y, self.adj, self.K, self.alpha, test_loader, 
                                                   self.device, other_idx_test, other_batch_size_test)



        # with torch.no_grad():
        #     # 提取模型输出特征
        #     embeddings = model.get_embeddings(X)  # 假设有一个 get_embeddings 方法获取节点特征
        #     embeddings = embeddings.cpu().numpy()  # 转换为 NumPy 数组

        # 使用 K-means 进行聚类
        embeddings=outputs
       # 设定聚类数目
        kmeans = KMeans(n_clusters=self.num_class_test, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        # 使用 t-SNE 降维
        tsne = TSNE(n_components=2, random_state=42)
        node_embeddings_2d = tsne.fit_transform(embeddings)

        # 可视化
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1], c=clusters, cmap='viridis', alpha=0.7)
        plt.title('Node Clustering Visualization')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.colorbar(scatter, label='Cluster')
       
       
       # 保存可视化图
        save_path = f'/root/SimTSC-main/SimTSC-main/visual_tsne_cluster/{dataset_name}/'
        os.makedirs(save_path, exist_ok=True)  # 自动创建目录
        plt.savefig(os.path.join(save_path, f'tsne_node_classification_epoch_{epoch}.png'))
        plt.show()
        printt('tsne 保存')
        plt.close()

    def visualize_node_classification(self, model, epoch):
        dataset_name = self.args.dataset

        # 将模型设置为评估模式
        model.eval()

        # 获取特征和邻接矩阵
        features = self.X.cpu().numpy()
        adj_matrix = self.adj.cpu().numpy()

        # 构建图
        # 根据 test_idx 提取对应的子矩阵
        new_adj_matrix = adj_matrix[np.ix_(self.test_idx, self.test_idx)]

        # 使用新的邻接矩阵构建图
        G = nx.from_numpy_array(new_adj_matrix)

        # 获取测试集的预测类别
        test_batch_size = min(self.batch_size // 2, len(self.test_idx))
        other_idx_test = np.array([i for i in range(len(self.X)) if i not in self.test_idx])
        other_batch_size_test = min(self.batch_size - test_batch_size, len(other_idx_test))
        
        test_dataset = Dataset(self.test_idx)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=1)
        
        outputs, predicted_classes = compute_classes(model, self.X, self.y, self.adj, self.K, self.alpha, test_loader, self.device, other_idx_test, other_batch_size_test)

        # 只选择测试集节点的标签和预测
        test_labels = self.y[self.test_idx].cpu().numpy()
        printt('测试集数量', len(test_labels))
        test_pred_classes = predicted_classes[:len(self.test_idx)]
        printt('test_pred_classes',len(test_pred_classes))

        # 可视化
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)  # 使用 spring 布局

        # 只绘制测试集节点
        nx.draw_networkx_nodes(G, pos, nodelist=range(len(test_pred_classes)),#self.test_idx, 
                               node_size=500,
                                node_color=test_pred_classes, cmap=plt.cm.jet, alpha=0.7)

        # 绘制边，只绘制连接测试集节点的边
        # edges_to_draw = [(u, v) for u, v in G.edges(self.test_idx) if u in self.test_idx and v in self.test_idx]
        # nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, alpha=0.5)

        edges_to_draw = [(u, v) for u, v in G.edges(range(len(test_pred_classes))) if u in 
                         range(len(test_pred_classes)) and v in range(len(test_pred_classes))]
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, alpha=0.5)


    

        # 添加标签
        # for i in self.test_idx:
        #     nx.draw_networkx_labels(G, pos, {i: test_labels[i]}, font_size=12)

         #添加标签
        for i, node in enumerate(self.test_idx):
            if node < len(test_labels):  # 确保不会超出 labels 的索引范围
                nx.draw_networkx_labels(G, pos, {node: test_labels[i]}, font_size=12)

        plt.title(f'Node Classification at Epoch {epoch}')
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.jet), label='Predicted Class')
        plt.axis('off')

        # 保存可视化图
        save_path = f'/root/SimTSC-main/SimTSC-main/visual/{dataset_name}/'
        os.makedirs(save_path, exist_ok=True)  # 自动创建目录
        plt.savefig(os.path.join(save_path, f'node_classification_epoch_{epoch}.png'))
        
        input()
        plt.close()


   
        dataset_name=self.args.dataset

        # 将模型设置为评估模式
        model.eval()

        # 获取特征和邻接矩阵
        features = self.X.cpu().numpy()
        adj_matrix = self.adj.cpu().numpy()

        # 构建图
        G = nx.from_numpy_array(adj_matrix)

        # 获取节点的预测类别
        test_batch_size = min(self.batch_size//2, len(self.test_idx))
        other_idx_test = np.array([i for i in range(len(self.X)) if i not in self.test_idx])
        other_batch_size_test = min(self.batch_size - test_batch_size, len(other_idx_test))
        test_dataset = Dataset(self.test_idx)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=1)
        outputs, predicted_classes = compute_class(model, self.X, self.y, self.adj, self.K, self.alpha, test_loader, self.device, other_idx_test, other_batch_size_test)
            


        # 获取每个节点的标签
        labels = self.y.cpu().numpy()

        # 可视化
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)  # 使用 spring 布局

        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_size=500,
                                node_color=predicted_classes, cmap=plt.cm.jet, alpha=0.7)

        # 绘制边
        nx.draw_networkx_edges(G, pos, alpha=0.5)

        # 添加标签
        for i in range(len(labels)):
            nx.draw_networkx_labels(G, pos, {i: labels[i]}, font_size=12)

        plt.title(f'Node Classification at Epoch {epoch}')
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.jet), label='Predicted Class')
        plt.axis('off')

        # 保存可视化图
        plt.savefig(f'/root/SimTSC-main/SimTSC-main/visual/{dataset_name}/node_classification_epoch_{epoch}.png')
        plt.close()

    def plot_loss_and_accuracy(self):
        # 绘制损失和准确率图
        plt.figure(figsize=(12, 5))

        # 绘制损失
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history, label='Loss', color='blue')
        plt.title('Loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()

        # 绘制准确率
        plt.subplot(1, 2, 2)
        plt.plot(self.acc_history, label='Accuracy', color='orange')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend()

        # 保存图像
        dataset_name = self.args.dataset  # 替换为实际的数据集名称
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_path = f'/root/SimTSC-main/SimTSC-main/accloss/{dataset_name}/'
        os.makedirs(save_path, exist_ok=True)  # 自动创建目录
        plt.savefig(os.path.join(save_path, f'train_lossandacc_{current_time}.png'))

    
        plt.close()


    def fit(self, model, X, y, train_idx, distances, K, alpha, test_idx=None, report_test=True, batch_size=16, epochs=500):#batch128
        self.test_idx=test_idx
        self.batch_size=batch_size
        
        self.K = K
        self.alpha = alpha
        print('===================fit')
        #20 (2858, 2858) 3 0.3
        print('textid', test_idx.shape)
        print('train_idx, distances, K, alpha', len(train_idx), distances.shape, K, alpha)
        
        train_batch_size = min(batch_size//2, len(train_idx))
        other_idx = np.array([i for i in range(len(X)) if i not in train_idx])
        print('other_idx', len(other_idx), )

        other_batch_size = min(batch_size - train_batch_size, len(other_idx))#108
        train_dataset = Dataset(train_idx)#
        #print('train_dataset', train_dataset.shape)

        #里面包含20个元素
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=1)
        print('batch_size//2, len(train_idx other_idx other_batch_size',
              batch_size//2, len(train_idx),other_idx.shape,other_batch_size)
        #64 2 (54,) 54  # 64 20 (2838,) 108
        if report_test:
            #
            print('======================report test')#testid 572
            test_batch_size = min(batch_size//2, len(test_idx))#64

            #排除测试机之外的
            other_idx_test = np.array([i for i in range(len(X)) if i not in test_idx])#2286
            other_batch_size_test = min(batch_size - test_batch_size, len(other_idx_test))#64
            test_dataset = Dataset(test_idx)#
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=1)
            print('test_batch_size, len(other_idx_test), other_batch_size_test, ', 
                  test_batch_size, len(other_idx_test), other_batch_size_test, )#64 2286 64

        self.adj = torch.from_numpy(distances.astype(np.float32))

        self.X, self.y = torch.from_numpy(X), torch.from_numpy(y)
        
        file_path = os.path.join(self.tmp_dir, str(uuid.uuid4()))

        #optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=4e-3)#-4

        best_acc = 0.0


        # Define optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=4e-3)

        # Using ReduceLROnPlateau to dynamically reduce learning rate
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True)

        
        torch.autograd.set_detect_anomaly(True)
        loss_fn = torch.nn.CrossEntropyLoss()
        # Training
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            for sampled_train_idx in train_loader:
                print('====================================train',sampled_train_idx)#tensor([33, 45])

                #从函数从 other_idx 2838 数组中随机选择 other_batch_size 108 个索引，且不允许重复选择（replace=False）
                sampled_other_idx = np.random.choice(other_idx, other_batch_size, replace=False)

                #加上确定好的加上随机选择的
                idx = np.concatenate((sampled_train_idx, sampled_other_idx))
                print('idx', len(idx))#128
                #[33 45 55 34 50 26 11  2 32 43 47 30  4 10 28 22 31 39 38  7 14 27 36 51
                    #  18 53 35 15  5 29 16 48 20 54  8 13 25 17 41 44  1 12 42 24  6 23 37 21
                    #  19  9 40 52  3  0 49 46]

                _X = self.X[idx].to(self.device)  # 使用 long() 确保 idx 是 LongTensor
                _y = self.y[sampled_train_idx.long()].to(self.device)  # 使用 long() 确保 sampled_train_idx 是 LongTensor
                _adj = self.adj[idx][:, idx]#.to(self.device)  # 使用 long() 确保 idx 是 LongTensor
                    #取出idx对应的样本+++ #取出训练idx用的标签 #取出行列对应的图结构元素
                #_X, _y, _adj = self.X[idx].to(self.device),  self.y[sampled_train_idx].to(self.device), self.adj[idx][:,idx]
                print('_X, _y, _adj',_X.shape, _y.shape, _adj.shape)
                #torch.Size([56, 1, 286]) torch.Size([2]) torch.Size([56, 56])
                #torch.Size([128, 3, 205]) torch.Size([20]) torch.Size([128, 128])

                outputs = model(_X, _adj, K, alpha)#.to(self.device)
                #outputs[:len(sampled_train_idx)] 表示只计算被采样的训练样本的损失,而不是整个批次。
                # 这可以提高训练效率。_y 则是这些被采样训练样本的真实类别标签。

                #=============
                loss = F.nll_loss(outputs[:len(sampled_train_idx)], _y)


                #=================
                
                #loss = loss_fn(outputs[:len(sampled_train_idx)], _y)

                #====================
                print('(outputs[:len(sampled_train_idx)], _y)',
                       outputs.shape, outputs[:len(sampled_train_idx)].shape, _y)#([56, 2])  tensor([[-0.5230, -0.8983],
      #  [-0.4983, -0.9354]], grad_fn=<SliceBackward0>)  tensor([0, 1])
      #torch.Size([128, 20]) torch.Size([20, 20])
                print('len(sampled_train_idx)', len(sampled_train_idx)) #20

                loss.backward()
                optimizer.step()

            model.eval()
            #所有的样本集合，训练集20个标签
            acc = compute_accuracy(model, self.X, self.y, self.adj, self.K, self.alpha, train_loader, self.device, other_idx, other_batch_size)
            
            # 保存损失和准确率
            self.loss_history.append(loss.item())
            self.acc_history.append(acc.cpu().numpy())

            
            if acc >= best_acc:
                best_acc = acc
                torch.save(model.state_dict(), file_path)
            if report_test:
                #用上了测试集的标签，572个


                #test_acc = compute_accuracy(model, self.X, self.y, self.adj, self.K, self.alpha, test_loader, self.device, other_idx_test, other_batch_size_test)
                test_acc, test_loss=self.test( model, test_idx, batch_size)
                # 记录准确率和损失
                self.accuracies.append(test_acc)  # 确保是浮点数
                self.losses.append(test_loss)  # 确保是浮点

                self.logger.log('--> Epoch {}: loss {:5.4f}; accuracy: {:5.4f}; best accuracy: {:5.4f}; test accuracy: {:5.4f}'.format(epoch, loss.item(), acc, best_acc, test_acc))
                scheduler.step(1 - test_acc)
            else:
                self.logger.log('--> Epoch {}: loss {:5.4f}; accuracy: {:5.4f}; best accuracy: {:5.4f}'.format(epoch, loss.item(), acc, best_acc))
                scheduler.step(1 - acc)

#=========================可视化
            # 每十个 epoch 进行一次可视化
            if epoch % 10 == 0:
                # 训练结束后进行节点聚类和可视化
                #self.visualize_node_clustering(model, X, y, distances, epoch)

                self.visualize_node_classification_tsne(model, epoch)

        # 训练结束后绘制损失和准确率图
       
        # self.save_accuracy_loss_plot(self.accuracies, self.losses)
        # self.plot_loss_and_accuracy()

#==========================    
        
        # Load the best model
        model.load_state_dict(torch.load(file_path))
        model.eval()
        os.remove(file_path)


        return model
    
    def test(self, model, test_idx, batch_size=128):
        test_batch_size = min(batch_size // 2, len(test_idx))
        other_idx_test = np.array([i for i in range(len(self.X)) if i not in test_idx])
        other_batch_size_test = min(batch_size - test_batch_size, len(other_idx_test))

        test_dataset = Dataset(test_idx)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=1)

        

        # 计算准确率和损失
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx in test_loader:
                sampled_other_idx = np.random.choice(other_idx_test, other_batch_size_test, replace=False)
                idx = np.concatenate((batch_idx, sampled_other_idx))
                _X, _y, _adj = self.X[idx].to(self.device), self.y[idx][:len(batch_idx)].to(self.device), self.adj[idx][:, idx]

                outputs = model(_X, _adj, self.K, self.alpha)

                # 计算损失（使用合适的损失函数）
                loss = F.nll_loss(outputs[:len(batch_idx)], _y)
               
                total_loss += loss.item()

                preds = outputs[:len(batch_idx)].max(1)[1].type_as(_y)
                _correct = preds.eq(_y).double()
                correct += _correct.sum()
                total += len(batch_idx)

            #     # 记录准确率和损失
            # accuracies.append(float(correct / total))  # 确保是浮点数
            # losses.append(float(total_loss / (len(accuracies))))  # 确保是浮点

        acc = correct / total
        avg_loss = total_loss / len(test_loader)

        # 保存图像
        # self.save_accuracy_loss_plot(accuracies, losses)

        return acc.item(), avg_loss


    

    def testo(self, model, test_idx, batch_size=128):
        test_batch_size = min(batch_size//2, len(test_idx))
        other_idx_test = np.array([i for i in range(len(self.X)) if i not in test_idx])
        other_batch_size_test = min(batch_size - test_batch_size, len(other_idx_test))
        test_dataset = Dataset(test_idx)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=1)
        acc = compute_accuracy(model, self.X, self.y, self.adj, self.K, self.alpha, test_loader, self.device, other_idx_test, other_batch_size_test)
        return acc.item()

    def save_accuracy_loss_plot(self,accuracies, losses):

            printt('accuracies, losses', len(accuracies), len(losses))
            printt("Type of accuracies:", type(accuracies))
            printt("Type of losses:", type(losses))

            plt.figure(figsize=(12, 6))

            # 绘制准确率
            plt.subplot(1, 2, 1)
            plt.plot(accuracies, label='Accuracy', color='blue')
            plt.title('Test Accuracy')
            plt.xlabel('Batch Index')
            plt.ylabel('Accuracy')
            plt.legend()

            # 绘制损失
            plt.subplot(1, 2, 2)
            plt.plot(losses, label='Loss', color='red')
            plt.title('Test Loss')
            plt.xlabel('Batch Index')
            plt.ylabel('Loss')
            plt.legend()

            plt.tight_layout()
            
            
            # 保存图像
            dataset_name = self.args.dataset  # 替换为实际的数据集名称
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

            save_path = f'/root/SimTSC-main/SimTSC-main/accloss/{dataset_name}/'
            os.makedirs(save_path, exist_ok=True)  # 自动创建目录
            plt.savefig(os.path.join(save_path, f'test_lossandacc_{current_time}.png'))

            plt.close()

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
    def __init__(self, input_size, nb_classes, args1, num_layers=1, n_feature_maps=512, dropout=0.5):
        super(SimTSC, self).__init__()
        self.num_layers = num_layers

        self.block_1 = ResNetBlock(input_size, n_feature_maps)
        self.block_2 = ResNetBlock(n_feature_maps, n_feature_maps)
        self.block_3 = ResNetBlock(n_feature_maps, n_feature_maps)

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
        hidden_channels  = 512 #128 #512#trial.suggest_categorical('hidden_channels', [16, 32, 64, 128])
        skip = False #True
        grid_size = 3 #trial.suggest_int('grid_size', 3, 5)
        num_classes=nb_classes
        num_features=512 #512 
        conv_type='gin'
        if self.num_layers == 1:
            self.kangin = GFASTKAN_Nodes( conv_type = conv_type,  num_layers = hidden_layers, num_features = num_features, hidden_channels= hidden_channels,
                        num_classes = num_classes, skip = skip, grid_size=grid_size)
            
        elif self.num_layers == 2:
            self.kangin1 = GFASTKAN_Nodes( conv_type = conv_type,  num_layers = hidden_layers, num_features = num_features, hidden_channels= hidden_channels,
                        num_classes = hidden_channels, skip = skip, grid_size=grid_size)
            self.kangin = GFASTKAN_Nodes( conv_type = conv_type,  num_layers = hidden_layers, num_features = num_features, hidden_channels= hidden_channels,
                        num_classes = num_classes, skip = skip, grid_size=grid_size)
        elif self.num_layers == 3:
            self.kangin1 = GFASTKAN_Nodes( conv_type = conv_type,  num_layers = hidden_layers, num_features = num_features, hidden_channels= hidden_channels,
                        num_classes = hidden_channels, skip = skip, grid_size=grid_size)
            self.kangin2 = GFASTKAN_Nodes( conv_type = conv_type,  num_layers = hidden_layers, num_features = num_features, hidden_channels= hidden_channels,
                        num_classes = hidden_channels, skip = skip, grid_size=grid_size)
            self.kangin = GFASTKAN_Nodes( conv_type = conv_type,  num_layers = hidden_layers, num_features = num_features, hidden_channels= hidden_channels,
                        num_classes = num_classes, skip = skip, grid_size=grid_size)
            
        self.linear1 = nn.Linear(1, 512)#128 512
        self.linear2 = nn.Linear(hidden_channels, nb_classes)#
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.linear3 = nn.Linear(nb_classes, nb_classes)
        self.linear4 = nn.Linear(45, 640)

    def forward(self, x, adj, K, alpha):

        

        print('===============================model',)
        print('x0',x.shape)# [8, 45, 2])

        # x=torch.transpose(x, 1, 2)
        # printt('x1',x.shape)#([8, 2, 45])

        # x=self.linear4(x)
        # printt('x3',x.shape)

        #============
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
        #============

        # x = x.permute(0, 2, 1)
        # printt('x1', x.shape)#[8, 2, 45])

        # x=self.linear4(x)
        # printt('x3',x.shape)#([8, 2, 640])

        # x = x.permute(0, 2, 1)
        # printt('x33', x.shape)#e([8, 640, 2])

        a=2
        if a==1:
            x = self.mamba(x)#x2 torch.Size([3, 512]) self.mamba(x)[0]
            print('x2',x.shape)#8, 2, 512])
        else:
            x = self.block_1(x)
            x = self.block_2(x)
            #x = self.block_3(x)
            x = x.permute(0, 2, 1)
            print('x33',x.shape)
        #====
        x = x.permute(0, 2, 1)
        print('x3',x.shape)#
        x = F.avg_pool1d(x, x.shape[-1]).squeeze()

        # x=torch.mean(x, dim=-1).squeeze() #max

        print('x4',x.shape)#8, 512])
        #===


        x=self.linear2(x)#
        #input()
        
        aa=2
        if aa==1:
            edge_index, edge_weight=graphtrans(adj)
            edge_index=edge_index.to(device)
            edge_weight=edge_weight.unsqueeze(1).to(device)
            edge_weight = self.linear1(edge_weight)
            print('edge_index, edge_weight',edge_index.shape, edge_weight.shape)#torch.Size([0]) torch.Size([0, 128])
            #input()
            if self.num_layers == 1:
                x = self.kangin(x, edge_index, edge_weight)#.to(device)
                #x = self.gc1(x, adj)
            elif self.num_layers == 2:
                
                x = self.kangin1(x, edge_index, edge_weight)
                print('ginx1', x.shape)#([8, 512])

                #x=self.linear2(x)

                x = self.kangin(x, edge_index, edge_weight)


                #====
                # x = F.relu(self.gc1(x, adj))
                # print('x5',x.shape)#5 torch.Size([4, 128])
                # x = F.dropout(x, self.dropout, training=self.training)
                # x = self.gc2(x, adj)
                # print('x6',x.shape)# torch.Size([4, 4])
            elif self.num_layers == 3:
                x = self.kangin1(x, edge_index, edge_weight)
                x = self.kangin2(x, edge_index, edge_weight)
                x = self.kangin(x, edge_index, edge_weight)

                #=====
                # x = F.relu(self.gc1(x, adj))
                # x = F.dropout(x, self.dropout, training=self.training)
                # x = F.relu(self.gc2(x, adj))
                # x = F.dropout(x, self.dropout, training=self.training)
                # x = self.gc3(x, adj)
            #===========
        print('gin x', x.shape) # [4, 4]) batch class

        #===========
        x = F.log_softmax(x, dim=1)

        #x = self.log_softmax(x)
        print('x5',x.shape)#x5 torch.Size([128, 20]) [4, 4])
        ##===========
       # x=self.linear3(x)
        #===========

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

