'''
# -*- coding: utf-8 -*-
@File    :   Solver.py
@Time    :   2024/10/27 17:21:27
@Author  :   Jiabing SUN 
@Version :   1.0
@Contact :   Jiabingsun777@gmail.com
@Desc    :   None
'''

# here put the import lib
import torch
import os
from os.path import join
from tqdm import tqdm
from ViT import Vit
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter
import torch.optim as optim


class Solver():
    def __init__(self, args) -> None:
        torch.autograd.set_detect_anomaly(True)
        self.args = args
        self.batch_size = self.args.batch_size
        self.num_epoch = self.args.num_epoch
        self.saveDir = "weight"
        self.task_name = self.args.task_name
        self.lr = self.args.lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.val_on_epoch = self.args.val_on_epoch
        self.resume = self.args.resume
        if self.args.model == 'vit':
            self.model = Vit()
        if not os.path.isdir(self.saveDir):
            os.makedirs(self.saveDir)
        self.model.cuda()

    def train(self):
        self.criterion = CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        transform = ToTensor()
        data = MNIST(root="./../data", train=True, download=True, transform=transform)
        train_size = int(len(data) * 5 / 6)  # 1:5 比例
        train_data = Subset(data, range(train_size))
        val_data = Subset(data, range(train_size, len(data)))
        num_workers = 0
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
        print(f"nums of train data: {len(train_data)}")
        print(f"nums of val data: {len(val_data)}")
        
        self.writer = SummaryWriter("log/" + self.task_name)

        start_epoch = 0
        best_acc = 0 
        if self.resume:
            best_name = self.task_name + '_best.pth'
            checkpoint = torch.load(join(self.saveDir, best_name))
            self.model.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['val_acc']
            print('load pretrained model---, start epoch at, ', start_epoch, ', star_psnr_val is: ', best_acc)
        for epoch in range(start_epoch, self.num_epoch):
            train_loss = self._train_vit(train_loader)
            if epoch % self.val_on_epoch == 0:
                val_acc, val_loss = self._validate(val_loader)
                print("Epoch {}/{}".format(epoch + 1, self.num_epoch))
                print(" val acc:\t\t{:.6f}".format(val_acc))
                print(" val loss:\t\t{:.6f}".format(val_loss))
                self.writer.add_scalar("loss/train_loss", train_loss, epoch)
                self.writer.add_scalar("metric/acc", val_acc, epoch)
                self.writer.flush()

                if best_acc < val_acc:
                    best_acc = val_acc
                    best_name = self.task_name + '_best.pth'
                    state = {'net': self.model.state_dict(), 'epoch': epoch, 'val_acc': val_acc}
                    torch.save(state, join(self.saveDir, best_name))
        self.writer.close()
        

    def _train_vit(self, train_loader):
        self.model.train()
        train_loss = 0
        for batch in tqdm(train_loader, position=0):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y) / len(x)
            train_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return train_loss

    def _validate_base(self):
        
        pass

    def _validate(self, val_loader):
        correct, total = 0, 0
        val_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, position=0):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y) / len(x)
                val_loss += loss
                correct += torch.sum(torch.argmax(y_hat, dim=1)==y).item()
                total += len(x)
        val_acc = correct / total
        return val_acc, val_loss