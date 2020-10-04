#Author:liulu
#Date:2020.10.04
#参考代码：https://github.com/sangyx/d2l-torch

#数据集：
    #Fashion-Mnist
    #训练集数量：60000
    #测试集数量：10000
'''
        ResNet-18模型复现
'''

import torch
import time
import sys

from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary

#加载数据集
def loadData(root, batch_size):
    # ResNet-18论文所用图片均为224*224，本程序将输入的高和宽从224降到96来简化计算
    transformer = transforms.Compose([transforms.Resize(96), transforms.ToTensor()])

    mnist_train = datasets.FashionMNIST(root=root, train=True, transform=transformer)
    mnist_test = datasets.FashionMNIST(root=root, train=False, transform=transformer)

    num_workers = 0 if sys.platform.startswith('win32') else 4

    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

#残差块，根据图：残差块结构
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        # 1x1卷积层用来将通道数改为想要的
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = self.conv1(X)
        Y = self.conv2(Y)
        if self.conv3:
            X = self.conv3(X)
        return  torch.relu(Y + X)

class MyResNet(nn.Module):
    def __init__(self):
        super(MyResNet, self).__init__()

        # ResNet的前两层跟之前介绍的GoogLeNet中的一样：在输出通道数为64、
        # 步幅为2的 7 x 7卷积层后接步幅为2的 3 x 3 的最大池化层。
        # 不同之处在于ResNet每个卷积层后增加的批量归一化层
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 与GoogLeNet中4个由Inception块组成的模块对应，ResNet使用4个由残差块组成的模块，
        # 每个模块使用若干个同样输出通道数的残差块
        self.b2 = nn.Sequential(
            Residual(64, 64),
            Residual(64, 64)
        )
        self.b3 = nn.Sequential(
            Residual(64, 128, use_1x1conv=True, stride=2),
            Residual(128, 128)
        )
        self.b4 = nn.Sequential(
            Residual(128, 256, use_1x1conv=True, stride=2),
            Residual(256, 256)
        )
        self.b5 = nn.Sequential(
            Residual(256, 512, use_1x1conv=True, stride=2),
            Residual(512, 512),
            # 加一个全局平均池化层调整大小
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # 与GoogLeNet一样，接上全连接层输出
        self.fc = nn.Linear(512*1*1, 10)

    def forward(self, X):
        X = self.b1(X)
        X = self.b2(X)
        X = self.b3(X)
        X = self.b4(X)
        X = self.b5(X)
        X = X.reshape(-1, 512*1*1)
        X = self.fc(X)
        return X

# 模型初始化
def params_init(model, init, **kwargs):
    def initializer(m):
        if isinstance(m, nn.Conv2d):
            init(m.weight.data, **kwargs)
            if m.bias is not None:
                m.bias.data.fill_(0)

        elif isinstance(m, nn.Linear):
            init(m.weight.data, **kwargs)
            if m.bias is not None:
                m.bias.data.fill_(0)

        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            if m.bias is not None:
                m.bias.data.fill_(0)

    model.apply(initializer)

#准确率的计算
def accrucy(data_iter, net, device):
    acc_sum, n = 0, 0
    with torch.no_grad():
        #将模型切换为预测模式
        net.eval()
        for X,y in data_iter:
            #数据复制到device
            X = X.to(device)
            y = y.to(device)
            acc_sum += (torch.argmax(net(X), dim=1) == y).sum().item()
            n += y.size(0)
        #将模型切换为训练模式
        net.train()
    return acc_sum / n

#训练
def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    print('training on ', device)
    #模型复制到device
    net.to(device)
    #设置为训练模式
    net.train()
    #采用交叉熵损失函数
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0, 0, 0
        start = time.time()
        for X,y in train_iter:
            #数据复制到device
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            #计算损失
            l = loss(y_hat, y)
            #反向传播
            l.backward()
            #更新参数
            optimizer.step()
            #梯度清零
            optimizer.zero_grad()
            train_l_sum += float(l)
            train_acc_sum += (torch.argmax(y_hat.data, dim=1)==y.data).sum().item()
            n += y.size(0)
        #计算在测试集上的准确率
        test_acc = accrucy(test_iter, net, device)
        print('epoch {0}, loss {1:.4f}, train_accrucy {2:.2%}, test_accrucy {3:.2%}, time span {4:.3f} sec'
              .format(epoch+1, train_l_sum/n, train_acc_sum/n, test_acc, time.time() - start))

if __name__ == '__main__':
    root = '~/dataset/'
    batch_size = 128
    train_iter, test_iter = loadData(root, batch_size)
    #如果有安装了cuda，则用cuda， 否则用cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MyResNet()
    num_epochs = 5
    # 学习率
    lr = 0.1
    # nn.init.xavier_uniform_是一种改进的服从均匀分布的初始化
    params_init(net, init=nn.init.xavier_uniform_)
    # 优化器采用随机梯度下降
    optimizer = optim.SGD(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
    # 显示网络结构，每一层的输出及参数个数
    summary(net, (1, 96, 96))