import concurrent.futures
import threading

import torch.nn as nn
import torch.nn.functional as F
import torch
from mamba_main.mamba_ssm import Mamba
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from print import print
from printt import printt
class EncoderLayer(nn.Module):
    def __init__(self, attention, attention_r, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.attention_r = attention_r
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.man = Mamba(
            d_model=11,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=2,  # Local convolution width
            expand=1,  # Block expansion factor)
        )
        self.man2 = Mamba(
            d_model=11,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=2,  # Local convolution width
            expand=1,  # Block expansion factor)
        )
        self.a = AttentionLayer(
                        FullAttention(False, 2, attention_dropout=0.1,
                                      output_attention=True), 11,1)
    def forward(self, x, attn_mask=None, tau=None, delta=None):
    # 反向注意力：
    # self.attention_r(x.flip(dims=[1]))：这里首先对输入 x 进行翻转（反向），然后将翻转后的输入传递给反向注意力机制 self.attention_r。
    # 翻转操作是通过 flip(dims=[1]) 实现的，这通常用于处理序列数据，以便从右到左进行注意力计算。
    # 反向注意力的目的是捕捉上下文信息的不同方面，特别是在处理时间序列或文本数据时，反向信息可能会提供额外的上下文。

    # 再次翻转：
    # .flip(dims=[1])：在反向注意力计算之后，再次翻转输出，以恢复原来的顺序。这样可以确保最终的输出与输入 x 的顺序一致。

    # 加法操作：
    # +：将正向和反向注意力的输出相加，形成 new_x。这种加法操作通常用于实现残差连接，使得网络能够更容易地学习到输入与输出之间的关系。

        print('mamba enc x',x.shape)#([4, 640, 512])
        new_x = self.attention(x) + self.attention_r(x.flip(dims=[1])).flip(dims=[1])
        print('newx', new_x.shape) #[4, 640, 512])
        attn = 1

        ####
        #============
        x = x + new_x

        x = self.norm1(x)
        #=============
        # y = x = self.norm1(x)
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))
        # return self.norm2(x + y), attn
        #===============
        #input()
        return x, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            print('=========================not none')
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            print('=========================self.conv_layers  none')
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)
        #input()
        if self.norm is not None:
            x = self.norm(x)

        return x, attns

