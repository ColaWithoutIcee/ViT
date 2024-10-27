'''
# -*- coding: utf-8 -*-
@File    :   main.py
@Time    :   2024/10/25 15:57:52
@Author  :   Jiabing SUN 
@Version :   1.0
@Contact :   Jiabingsun777@gmail.com
@Desc    :   None
'''

# here put the import lib
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from ViT import Vit


transform = ToTensor()
train_dataset = MNIST(root="./../data", train=True, download=True, transform=transform)
test_dataset = MNIST(root="./../data", train=True, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epoches = 50
lr = 0.001

model = Vit()
model.cuda()
optimizer = Adam(model.parameters(), lr=lr)
criterion = CrossEntropyLoss()
model.train()
for epoch in range(n_epoches):
    train_loss = 0.0
    for batch in tqdm(train_loader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y) / len(x)

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{n_epoches} loss: {train_loss:.2f}")

 # Test loop
correct, total = 0, 0
test_loss = 0.0
model.eval()
with torch.no_grad():
    for batch in test_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y) / len(x)
        test_loss += loss
        correct += torch.sum(torch.argmax(y_hat, dim=1) == y).item()
        total += len(x)
    print(f"Test loss: {test_loss:.2f}")
    print(f"Test accuracy: {correct / total * 100:.2f}%")
