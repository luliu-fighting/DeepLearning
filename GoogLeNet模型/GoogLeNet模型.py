#Author:liulu
#Date:2020.10.03
#参考代码：https://github.com/sangyx/d2l-torch

#数据集：
    #Fashion-Mnist
    #训练集数量：60000
    #测试集数量：10000



import torch
import time
import sys

from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import  summary

#加载数据集
def loadData(root, batch_size):
    #GoogLeNet论文所用图片均为224*224，本程序将输入的高和宽从224降到96来简化计算
    transformer = transforms.Compose([transforms.Resize(96), transforms.ToTensor()])
    mnist_train = datasets.FashionMNIST(root=root, train=True, transform=transformer)
    mnist_test = datasets.FashionMNIST(root=root, train=False, transform=transformer)

    num_workers = 0 if sys.platform.startswith('win32') else 4

    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

#Inception块
class Inception(nn.Module):
    # c0为输入通道数, c1 - c4为每条线路里的层的输出通道数
    # 4条线路都使用了合适的填充来使输入与输出的高和宽一致
    def __init__(self, c0, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 线路1，单1 x 1卷积层
        self.p1 = nn.Sequential(
            nn.Conv2d(c0, c1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2 = nn.Sequential(
            nn.Conv2d(c0, c2[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # 线路3， 1 x 1卷积层后接5 x 5卷积层
        self.p3 = nn.Sequential(
            nn.Conv2d(c0, c3[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        # 线路4， 3 x 3最大池化层后接1 x 1卷积层
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(c0, c4, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.p1(x)
        x2 = self.p2(x)
        x3 = self.p3(x)
        x4 = self.p4(x)
        # 在通道维上连结输出
        return torch.cat((x1, x2, x3, x4), dim=1)

#GoogLeNet在主体卷积部分中使用5个模块（block），每个模块之间使用步幅为2的$3\times 3$最大池化层来减小输出高宽。
class MyGoogLeNet(nn.Module):
    def __init__(self):
        super(MyGoogLeNet, self).__init__()
        # 第一模块使用一个64通道的7 x 7卷积层。
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 第二模块使用2个卷积层
        self.b2 = nn.Sequential(
            # 首先是64通道的1 x 1卷积层
            nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(inplace=True),  # inplace-选择是否进行覆盖运算
            # 然后是将通道增大3倍的3 x 3卷积层。它对应Inception块中的第二条线路。
            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 第三模块串联2个完整的Inception块。
        self.b3 = nn.Sequential(
            # 第一个Inception块的输出通道数为 64+128+32+32=256，
            # 其中4条线路的输出通道数比例为 64:128:32:32=2:4:1:1。
            # 其中第二、第三条线路先分别将输入通道数减小至 96/192=1/2 和 16/192=1/12 后，再接上第二层卷积层。
            Inception(192, 64, (96, 128), (16, 32), 32),
            # 第二个Inception块输出通道数增至 128+192+96+64=480，
            # 每条线路的输出通道数之比为 128:192:96:64 = 4:6:3:2。
            # 其中第二、第三条线路先分别将输入通道数减小至 128/256=1/2 和 32/256=1/8 。
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 第四模块更加复杂。它串联了5个Inception块。
        # 每个Inception块的线路的通道数分配和第三模块中的类似，
        # 首先含3 x 3卷积层的第二条线路输出最多通道，其次是仅含1 x 1卷积层的第一条线路，
        # 之后是含5 x 5卷积层的第三条线路和含3 x 3最大池化层的第四条线路。
        # 其中第二、第三条线路都会先按比例减小通道数。这些比例在各个Inception块中都略有不同。
        self.b4 = nn.Sequential(
            # 输出通道数 192+208+48+64=512
            Inception(480, 192, (96, 208), (16, 48), 64),
            # 输出通道数 160+224+64+64=512
            Inception(512, 160, (112, 224), (24, 64), 64),
            # 输出通道数 128+256+64+64=512
            Inception(512, 128, (128, 256), (24, 64), 64),
            # 输出通道数 112+288+64+64=528
            Inception(512, 112, (144, 288), (32, 64), 64),
            # 输出通道数 256+320+128+128=832
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 第五模块有两个Inception块。其中每条线路的通道数的分配思路和第三、第四模块中的一致，
        # 只是在具体数值上有所不同。需要注意的是，第五模块的后面紧跟输出层，
        # 该模块同NiN一样使用全局平均池化层来将每个通道的高和宽变成1。
        self.b5 = nn.Sequential(
            # 输出通道数 256+320+128+128=832
            Inception(832, 256, (160, 320), (32, 128), 128),
            # 输出通道数 384+384+128+128=1024
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 最后我们将输出变成二维数组后接上一个输出个数为标签类别数的全连接层。
        self.output = nn.Linear(1024 * 1 * 1, 10)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = x.reshape(-1, 1024 * 1 * 1)
        x = self.output(x)
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
    net = MyGoogLeNet()
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