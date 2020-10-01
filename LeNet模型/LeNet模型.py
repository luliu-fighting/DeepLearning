#Author:liulu
#Date:2020.10.01
#参考代码：https://github.com/sangyx/d2l-torch

#数据集：
    #Fashion-Mnist
    #训练集数量：60000
    #测试集数量：10000
import torch
import time

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn,optim
from torchsummary import summary

#加载数据集
def loadData(root, batch_size):
    print('start to load data')
    mnist_train = datasets.FashionMNIST(root=root, train=True, transform=transforms.ToTensor())
    mnist_test = datasets.FashionMNIST(root=root, train=False, transform=transforms.ToTensor())

    num_workers = 0

    # 每次读取一个样本数为batch_size的小批量数据，shuffle为True表示打乱顺序读取，即随机读取
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print('completed!')
    return train_iter, test_iter

class MyLeNet(nn.Module):
    def __init__(self):
        super(MyLeNet, self).__init__()
        #卷积部分
        self.conv = nn.Sequential(  #输入图片大小为1*28*28
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),  #经过第一个卷积层，6*24*24
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),   #最大池化后，6*12*12
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), #经过第二个卷积层，16*8*8
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2)   #最大池化后，16*4*4
        )
        #全连接部分
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)  #输出个数为10，表示对每一个类别的预测概率
        )
    #前向传播
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(-1, 16*4*4)
        x = self.fc(x)
        return x

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

if __name__ == '__main__':
    root = '~/dataset/'
    batch_size = 256
    train_iter, test_iter = loadData(root, batch_size)
    #如果有安装了cuda，则用cuda， 否则用cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MyLeNet()
    num_epochs = 10
    #学习率
    lr = 1.5
    #nn.init.xavier_uniform_是一种改进的服从均匀分布的初始化
    params_init(net, init=nn.init.xavier_uniform_)
    #优化器采用随机梯度下降
    optimizer = optim.SGD(net.parameters(), lr=lr)
    train(net, train_iter, test_iter,batch_size, optimizer, device, num_epochs)
    #显示网络结构，每一层的输出及参数个数
    summary(net, (1, 28, 28))

