from node_classification.ekan import KAN as eKAN
import sys
import os
from os import path
d = path.dirname(__file__)  # 获取当前路径
parent_path = os.path.dirname(d)  # 获取上一级路径
sys.path.append(parent_path)    # 如果要导入到包在上一级
from torch.nn import Module
from fast_kan.fastkan.fastkan import FastKAN
import torch

from torch.nn import Sequential
from torch_geometric.nn import GINConv, GCNConv, GINEConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from print import print
from printt import printt


class GCKANLayer(torch.nn.Module):
    def __init__(self, in_feat:int,
                 out_feat:int,
                 grid_size:int=4,
                 spline_order:int=3):
        super(GCKANLayer, self).__init__()
        self.kan = eKAN([in_feat, out_feat], grid_size=grid_size, spline_order=spline_order)


    def forward(self, X, A_hat_normalized):
        return self.kan(A_hat_normalized @ X)

class GIKANLayer(GINConv):
    def __init__(self, in_feat:int,
                 out_feat:int,
                 grid_size:int=4,
                 spline_order:int=3):
        kan = eKAN([in_feat, out_feat], grid_size=grid_size, spline_order=spline_order)
        GINConv.__init__(self, kan)
        #GINEConv.__init__(self, kan)


class GCFASTKANLayer(torch.nn.Module):###
    def __init__(self, in_feat:int,
                 out_feat:int,
                 grid_size:int=4):
        super(GCFASTKANLayer, self).__init__()
        self.kan = FastKAN([in_feat, out_feat], num_grids=grid_size)


    def forward(self, X, A_hat_normalized):
        return self.kan(A_hat_normalized @ X)

class GIFASTKANLayer(GINEConv):###GINConv
    def __init__(self, in_feat:int,
                 out_feat:int,
                 grid_size:int=4):
        kan = FastKAN([in_feat, out_feat], num_grids=grid_size)
        #GINConv.__init__(self, kan)
        GINEConv.__init__(self, kan )


class GNN_Nodes(torch.nn.Module):
    def __init__(self,  conv_type :str,
                 num_layers:int,
                 num_features:int,
                 hidden_channels:int,
                 num_classes:int,
                 skip:bool = True):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers-1):
            if i ==0:
                if conv_type == "gcn":
                    self.convs.append(GCNConv(num_features, hidden_channels))
                else:
                    self.convs.append(GINConv(Sequential(
                    torch.nn.Linear(num_features, hidden_channels),
                    torch.nn.ReLU() , Linear( hidden_channels, hidden_channels ),
                    torch.nn.ReLU()) ))

            else:
                if conv_type == "gcn":
                    self.convs.append(GCNConv(hidden_channels, hidden_channels))
                else:
                    self.convs.append(GINConv(Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU() , Linear( hidden_channels, hidden_channels ),
                    torch.nn.ReLU()) ))

        self.skip = skip

        if self.skip:
            if conv_type == "gcn":
                self.conv_out =  GCNConv(num_features+(num_layers-1)*hidden_channels, num_classes)
            else:
                self.conv_out = GINConv(Sequential(
                torch.nn.Linear(num_features+(num_layers-1)*hidden_channels, hidden_channels),
                torch.nn.ReLU() , torch.nn.Linear( hidden_channels, num_classes ),
                torch.nn.ReLU()) )
        else:
            if conv_type == "gcn":
                self.conv_out =  GCNConv(hidden_channels, num_classes)
            else:
                self.conv_out = GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU() , torch.nn.Linear( hidden_channels, num_classes ),
                torch.nn.ReLU()) )


    def forward(self, x: torch.tensor , edge_index: torch.tensor):
        l = []
        l.append(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.nn.functional.relu(x)
            l.append(x)

        if self.skip:
            x = torch.cat(l, dim=1)

        x = self.conv_out(x, edge_index)
        x = torch.nn.functional.relu(x)
        return x




class GKAN_Nodes(torch.nn.Module):
    def __init__(self, conv_type :str,
                 num_layers:int,
                 num_features:int,
                 hidden_channels:int,
                 num_classes:int,
                 skip:bool = True,
                 grid_size:int = 4,
                 spline_order:int = 3):
        super().__init__()

        dic = {"gcn": GCKANLayer , "gin": GIKANLayer}
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers-1):
            if i ==0:
                self.convs.append(dic[conv_type](num_features, hidden_channels, grid_size, spline_order))
            else:
                self.convs.append(dic[conv_type]( hidden_channels, hidden_channels, grid_size, spline_order))

        self.skip = skip

        if self.skip:
            self.conv_out = dic[conv_type](num_features+(num_layers-1)*hidden_channels,
                                       num_classes, grid_size, spline_order)
        else:
            self.conv_out = dic[conv_type](hidden_channels, num_classes, grid_size, spline_order)


    def forward(self, x: torch.tensor, edge_index: torch.tensor):
        l = []
        l.append(x)
        for conv in self.convs:

            x = conv(x, edge_index)
            l.append(x)

        if self.skip:
            x = torch.cat(l, dim=1)

        x = self.conv_out(x,edge_index)
        return x

class GFASTKAN_Nodes(torch.nn.Module):
    def __init__(self, conv_type :str,
                 num_layers:int,
                 num_features:int,
                 hidden_channels:int,
                 num_classes:int,
                 skip:bool = True,
                 grid_size:int = 4):
        super().__init__()

        dic = {"gcn": GCFASTKANLayer , "gin": GIFASTKANLayer}
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers-1):
            if i ==0:
                self.convs.append(dic[conv_type](num_features, hidden_channels, grid_size))
            else:
                self.convs.append(dic[conv_type]( hidden_channels, hidden_channels, grid_size))

        self.skip = skip

        if self.skip:
            self.conv_out = dic[conv_type](num_features+(num_layers-1)*hidden_channels,
                                       num_classes, grid_size)
        else:
            self.conv_out = dic[conv_type](hidden_channels, num_classes, grid_size)


    def forward(self, x: torch.tensor, edge_index: torch.tensor,edge_weight: torch.tensor):#
        print('x',x.shape)##[2708, 1433] torch.Size([128, 512])
        print('edge_index',edge_index.shape)##[2, 10556] torch.Size([2, 384])
        # 该数据集共2708个样本点，每个样本点都是一篇科学论文，所有样本点被分为8个类别，类别分别是
        # 1）基于案例；2）遗传算法；3）神经网络；4）概率方法；5）强化学习；6）规则学习；7）理论
        # 每篇论文都由一个1433维的词向量表示，所以，每个样本点具有1433个特征。词向量的每个元素都对应一个词，
        # 且该元素只有0或1两个取值。取0表示该元素对应的词不在论文中，取1表示在论文中。所有的词来源于一个具有1433个词的字典。
        # 每篇论文都至少引用了一篇其他论文，或者被其他论文引用，也就是样本点之间存在联系，
        # 没有任何一个样本点与其他样本点完全没联系。
        # 如果将样本点看做图中的点，则这是一个连通的图，不存在孤立点。

        l = []
        l.append(x)
        for conv in self.convs:
            print('x, edge_index, edge_weight', x.shape,edge_index.shape,edge_weight.shape)
            x = conv(x, edge_index, edge_weight)
            l.append(x)

        if self.skip:
            x = torch.cat(l, dim=1)

        #torch.Size([128, 512]) torch.Size([2, 384]) torch.Size([384])
        print('x, edge_index, edge_weight', x.shape,edge_index.shape,edge_weight.shape)
        x = self.conv_out(x, edge_index, edge_weight)
        print('gin x1', x.shape)#[4, 4])
        return x