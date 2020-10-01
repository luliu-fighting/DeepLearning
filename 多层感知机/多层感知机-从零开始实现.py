#Author:liulu
#Date:2020.09.11
#参考代码：https://github.com/sangyx/d2l-torch

#数据集：Fashion-MINST数据集
'''
此程序与‘softmax回归-从零开始实现’代码大部分相同，不同之处只是将模型的初始化集成在了一个函数中。
本程序是多层感知机的实现，相当于2层神经网络，隐藏层的神经元个数设置为256，
激活函数使用的是Relu，并未使用pytorch自带的Relu，而是使用torch的clamp函数实现的；
为了得到更好的数值稳定性，直接使用了nn提供的包括softmax运算和交叉熵损失计算的交叉熵损失函数。

其次，在训练过程中发现，学习率设置为0.5时，需要10+个epoch才能达到预期的正确率，于是将学习率设置为50，
此时只需4个epoch即可达到比较好的结果。出现上述现象的原因应该是：自定义的梯度下降法在每次更新参数时都会除
小批量样本的数量，所以相对的使学习率变小。若使用torch中自带的梯度下降函数则不会出现此现象。
'''
import sys
import torch
import time

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn


#加载数据集
def load_data_fashion_mnist(root, batch_size):
    minst_train = datasets.FashionMNIST(root=root, train=True, download=False, transform=transforms.ToTensor())
    minst_test = datasets.FashionMNIST(root=root, train=False, download=False, transform=transforms.ToTensor())

    num_workers = 0 if sys.platform.startswith('win32') else 4

    train_iter = DataLoader(minst_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(minst_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_iter, test_iter


#初始化模型参数
def init_params():
    num_inputs = 784
    #隐藏层的神经元个数
    num_hiddens = 256
    num_outputs = 10
    w1 = torch.normal(mean=torch.zeros(num_inputs, num_hiddens), std=0.01)
    b1 = torch.zeros(num_hiddens)
    w2 = torch.normal(mean=torch.zeros(num_hiddens, num_outputs), std=0.01)
    b2 = torch.zeros(num_outputs)
    params = [w1, b1, w2, b2]
    for param in params:
        param.requires_grad_()
    return w1, b1, w2, b2


#定义激活函数
def relu(X):
    #clamp(X,min=0)：将X中小于零的数据变为0
    return torch.clamp(X, min=0)


#定义模型
def net(X, num_inputs=784):
    X = X.reshape(-1, num_inputs)
    H = relu(torch.mm(X, w1) + b1)
    Output = torch.mm(H, w2) + b2
    return Output


#定义损失函数
    #分开定义softmax运算和交叉熵损失函数可能会造成数值不稳定
    #所以为了得到更好的数值稳定性，直接使用了nn提供的包括softmax运算和交叉熵损失计算的函数。
    #loss = nn.CrossEntropyLoss()

#定义优化算法
def sgd(params, lr, batch_size):
    #小批量随机梯度下降
    for param in params:
        param.data.sub_(lr*param.grad.data / batch_size)
        #每次更新完参数要对梯度进行清零
        param.grad.data.zero_()


#训练模型
def train(net, train_iter, loss, num_epoch, batch_size, params=None, lr=None, optimizer=None):

    for epoch in range(num_epoch):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X,y in train_iter:
            #获取每个样本对每种类别的预测概率
            y_hat = net(X)
            #计算小批量样本的交叉熵损失之和
            l = loss(y_hat, y).sum()
            #反向传播，自动求梯度
            l.backward()
            #判断是否已指定优化器，没有则使用自己定义的随机梯度下降算法
            if optimizer is None:
                sgd(params=params, lr=lr, batch_size=batch_size)
            else:
                optimizer.step()
                optimizer.zero_grad()
            #求所有小批量样本的损失之和，即本次迭代的损失
            #这里用item()将数据由tensor变为标量数值
            train_l_sum += l.data.item()
            #求所有样本中分类正确的样本数之和
            train_acc_sum += (y_hat.data.argmax(dim=1) == y).sum()
            #所用到的样本总数
            n += y.size(0)
        print("epoch:{0}, loss:{1:.4f}, train_accrucy:{2:.3f}".format(epoch+1, train_l_sum / n, train_acc_sum / n))


#获取图片对应的文本标签
def get_text_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

    return [text_labels[int(i)] for i in labels]


#测试模型
def test(net, test_iter):
    test_acc_sum, n = 0, 0
    #不需要更新参数，所以不需要记录梯度信息
    with torch.no_grad():
        #取第一个小批量数据，为了展示预测结果和真实标签的对比
        for X, y in test_iter:
            break
        #真实标签
        true_labels = get_text_labels(y)
        #预测的标签
        predict_labels = get_text_labels(net(X).data.argmax(dim=1))
        #打印输出，这里只打印了10个输出
        print(true_labels[0:9])
        print(predict_labels[0:9])

        #计算测试集所用样本的分类正确率
        for X,y in test_iter:
            test_acc_sum += (net(X).data.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
    print("test accuracy = %.4f" % (test_acc_sum / n))

#主程序
if __name__ == "__main__":
    start = time.time()
    #设置小批量数据量大小
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(root="~/dataset/", batch_size=batch_size)
    w1, b1, w2, b2 = init_params()
    #设置迭代次数
    num_epoch = 10
    #设置学习率，因为要除小批量样本的数量，所以设置的大一点
    lr = 50
    #训练
    train(net, train_iter, loss=nn.CrossEntropyLoss(), num_epoch=num_epoch,
          batch_size=batch_size, params=[w1, b1, w2, b2], lr=lr)
    #耗时
    print("time span %f" % (time.time()-start))
    test(net, test_iter)