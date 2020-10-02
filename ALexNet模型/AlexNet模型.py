#Author:liulu
#Date:2020.10.02
#参考代码：https://github.com/sangyx/d2l-torch

#数据集：
    #Fashion-Mnist
    #训练集数量：60000
    #测试集数量：10000

'''
本程序实现的是稍微简化过的AlexNet。
    AlexNet的代码与LeNet的代码差别只在网络结构上，训练、准确率的计算、模型初始化的代码均相同。

    AlexNet论文中使用ImageNet数据集，但因为ImageNet数据集训练时间较长，本程序仍用前面的Fashion-MNIST数据集来训练AlexNet。
读取数据的时候额外做了一步将图像高和宽扩大到AlexNet使用的图像高和宽224。

    因为本人电脑比较老旧，运行速度慢，且GPU内存很小，于是将batch_size调整为了128，学习率设为0.01.
即使这样，每个epoch都要跑十几分钟。
'''

import torch
import time
import sys

from torch import nn, optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torchsummary import summary

#加载数据集
def loadData(root, batch_size):
    #Fashion-Mnist数据集图片大小均为28*28，而AlexNet模型需要输入大小为224*24，所以需要resize
    transformer = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
    mnist_train = datasets.FashionMNIST(root=root, train=True, transform=transformer)
    mnist_test = datasets.FashionMNIST(root=root, train=False, transform=transformer)

    num_workers = 0 if sys.platform.startswith('win32') else 4

    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.conv = nn.Sequential(    #输入1*224*224
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4), #96*54*54
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), #96*26*26
            nn.Conv2d(96, 256, kernel_size=5, padding=2), #256*26*26
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  #256*12*12
            nn.Conv2d(256, 384, kernel_size=3, padding=1), #384*12*12
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  #384*12*12
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  #256*12*12
            nn.MaxPool2d(kernel_size=3, stride=2)       #256*5*5
        )

        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            #为了防止过拟合，采用了丢弃法，让一些神经元随机失活
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(-1, 256*5*5)
        x = self.fc(x)
        return x

#模型初始化
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
    net = MyAlexNet()
    num_epochs = 5
    #学习率
    lr = 0.01
    #nn.init.xavier_uniform_是一种改进的服从均匀分布的初始化
    params_init(net, init=nn.init.xavier_uniform_)
    #优化器采用随机梯度下降
    optimizer = optim.SGD(net.parameters(), lr=lr)
    train(net, train_iter, test_iter,batch_size, optimizer, device, num_epochs)
    #显示网络结构，每一层的输出及参数个数
    summary(net, (1, 224, 224))