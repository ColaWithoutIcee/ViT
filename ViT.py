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


class TransformerEncoder(nn.Module):
    
    def __init__(self, emb_size):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.MSA = MultiHeadAttention(emb_size)
        self.mlp = Mlp(emb_size, emb_size*2)
    
    def forward(self, x):
        x1 = self.norm(x)
        att_out = self.MSA(x)
        x2 = x + att_out
        x3 = self.norm(x2)
        mlp_out = self.mlp(x3)
        encoder_out = x2 + mlp_out
        print(f"encoder_out size {encoder_out.shape}")
        return encoder_out

class ClassificationHead(nn.Module):

    def __init__(self, emb_size: int=16, n_class: int=10):
        super().__init__()
        self.linear = nn.Linear(emb_size, n_class)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        print(f"cls head output: {x.shape}")
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
        print(f"output:{out.shape}")
        return out


class Vit(nn.Module):

    def __init__(self, image, patch_size, encoder_depth = 7):
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
        # transformer encoders 
        self.encoders = nn.Sequential(*[TransformerEncoder(emb_size=16) for _ in range(encoder_depth)])
        # classification head
        self.cls_head = ClassificationHead()
    
    def forward(self, images):
        b, c, h, w = images.shape
        patches = images.reshape(b, c*(h // self.patch_size[0])*(w // self.patch_size[1]), self.input_d)
        tokens = self.liear_pro(patches)
        tokens = torch.cat((self.cls_token, tokens), dim=1)
        tokens += self.pos_embbing
        out = self.encoders(tokens)
        print(f"encoders out: {out.shape}")
        # get the classification token only
        out = out[:,0,:]
        # classification
        out = self.cls_head(out)
        return out