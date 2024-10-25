'''
# -*- coding: utf-8 -*-
@File    :   ViT.py
@Time    :   2024/10/22 15:00:44
@Author  :   Jiabing SUN 
@Version :   1.0
@Contact :   Jiabingsun777@gmail.com
@Desc    :   None
'''

# here put the import lib
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = 4, emb_size: int = 16, img_size: int = 28):
        super().__init__()
        self.patch_size = patch_size
        self.input_d = patch_size ** 2
        self.in_channels = in_channels
        # self.get_patches = Rearrange()
        self.liear_pro = nn.Sequential(
            # 使用卷积层来代替线性层
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange("b e h w -> b (h w) e")
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn(1 + (img_size//patch_size) ** 2, emb_size))
    
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.liear_pro(x)
        cls_token = repeat(self.cls_token, "() n e -> b n e", b = b)
        x = torch.cat((cls_token, x), dim = 1)
        x += self.positions
        return x


class Mlp(nn.Module):
    def __init__(self, input_features, hidden_features=None, output_features=None, act_layer=nn.GELU(), dropout=0.):
        super().__init__()
        output_features = output_features or input_features
        hidden_features = hidden_features or input_features
        self.fc1 = nn.Linear(input_features, hidden_features)
        self.act_layer = act_layer
        self.fc2 = nn.Linear(hidden_features, output_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoder(nn.Module):
    
    def __init__(self, emb_size: int=16, n_heads: int=4):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.MSA = MultiHeadAttention(emb_size, n_heads)
        self.mlp = Mlp(emb_size, emb_size*2)
    
    def forward(self, x):
        x1 = self.norm(x)
        att_out = self.MSA(x1)
        x2 = x + att_out
        x3 = self.norm(x2)
        mlp_out = self.mlp(x3)
        encoder_out = x2 + mlp_out
        return encoder_out

class ClassificationHead(nn.Module):

    def __init__(self, emb_size: int=16, n_class: int=10):
        super().__init__()
        self.linear = nn.Linear(emb_size, n_class)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, emb_size: int = 16, n_heads: int = 4, dropout: float = 0.):
        super().__init__()
        self.emb_size = emb_size
        self.n_heads = n_heads
        # cal Q K V mats
        self.qkv = nn.Linear(emb_size, emb_size*3)
        self.scaling = (emb_size // n_heads) ** (1/2)
        self.att_drop = nn.Dropout(dropout)
        self.pro = nn.Linear(emb_size, emb_size)
        
    def forward(self, x):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> qkv b h n d", qkv=3, h=self.n_heads)
        Queries, Keys, Values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum("bhqd, bhkd -> bhqk", Queries, Keys)
        energy = energy / self.scaling
        att = F.softmax(energy, dim=-1)
        att = torch.einsum("bhal, bhlv -> bhav", att, Values)
        out = rearrange(att, "b h n d -> b n (h d)")
        out =  self.pro(out)
        return out


class Vit(nn.Module):

    def __init__(self, in_channels: int=1, patch_size: int=4, emb_size: int=16, img_size: int=28, n_heads: int=4, encoder_depth = 7):
        super().__init__()
        # patch embedding
        self.patch_emb = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        # transformer encoders 
        self.encoders = nn.Sequential(*[TransformerEncoder(emb_size=emb_size, n_heads=n_heads) for _ in range(encoder_depth)])
        # classification head
        self.cls_head = ClassificationHead(emb_size)
    
    def forward(self, x):
        tokens = self.patch_emb(x)
        out = self.encoders(tokens)
        # get the classification token only
        out = out[:,0,:]
        # classification
        out = self.cls_head(out)
        return out