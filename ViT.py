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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


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
        print(f"qkv1 shape: {x.shape}")
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> qkv b h n d", qkv=3, h=self.n_heads)
        print(f"qkv2 shape {qkv.shape}")
        Queries, Keys, Values = qkv[0], qkv[1], qkv[2]
        print(f"Queries: {Queries.shape}")
        print(f"Keys: {Keys.shape}")
        print(f"Values: {Values.shape}")
        energy = torch.einsum("bhqd, bhkd -> bhqk", Queries, Keys)
        print(f"Q @ K^T: {energy.shape}")
        energy = energy / self.scaling
        att = F.softmax(energy, dim=-1)
        att = torch.einsum("bhal, bhlv -> bhav", att, Values)
        print(f"attention: {att.shape}")
        out = rearrange(att, "b h n d -> b n (h d)")
        out =  self.pro(out)
        print(f"output:{out.shape}")


class Vit(nn.Module):

    def __init__(self, image, patch_size):
        super().__init__()
        self.batch, self.channel, self.image_h, self.image_w  = image.shape
        self.patch_size = patch_size
        assert (self.image_h % patch_size[0] == 0) and (self.image_w % patch_size[1] == 0),  "patch_size invalid"
        self.input_d = patch_size[0] * patch_size[1]
        self.liear_pro = nn.Linear(self.input_d, self.input_d)
        # Classification token
        self.cls_token = nn.Parameter(torch.randn(self.batch, 1, self.input_d))
        # Position embedding 
        self.pos_embbing = nn.Parameter(torch.randn(1 + self.channel*(self.image_h // self.patch_size[0])*(self.image_w // self.patch_size[1]), self.input_d))
    
    def forward(self, images):
        b, c, h, w = images.shape
        patches = images.reshape(b, c*(h // self.patch_size[0])*(w // self.patch_size[1]), self.input_d)
        tokens = self.liear_pro(patches)
        tokens = torch.cat((self.cls_token, tokens), dim=1)
        tokens += self.pos_embbing
        return tokens