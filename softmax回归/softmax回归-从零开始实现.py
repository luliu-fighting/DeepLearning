#Author:liulu
#Date:2020.09.09
#参考代码：https://github.com/sangyx/d2l-torch

'''
数据集：Fashion-MINST数据集
数据集简介：Fashion-MNIST中一共包括了10个类别，分别为t-shirt（T恤）、trouser（裤子）、
pullover（套衫）、dress（连衣裙）、coat（外套）、sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）、
bag（包）和ankle boot（短靴）。
训练集中和测试集中的每个类别的图像数分别为6,000和1,000。因为有10个类别，所以训练集和测试集的样本数分别为60,000和10,000。

'''
import sys
import torch
import time

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


#加载Fashion_MINST数据集
def load_data_fashion_minst(root, batch_size, download = False, transformer=transforms.ToTensor()):
    #加载数据时使用了transforms.ToTensor（）, 它会将一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[通道数,高,宽]，取值范围是[0,1.0]的torch.FloatTensor。因为数据集中是灰度图像，所以通道数为1。
    #使用transforms.ToPILImage可以将shape为(C,H,W)的Tensor或shape为(H,W,C)的numpy.ndarray转换成PIL.Image，值不变。

    #读取训练集和数据集，因为已经离线下载了数据集，所以这里的download设为False，root是数据集所在的目录
    minst_train = datasets.FashionMNIST(root=root, train=True, transform=transformer, download=False)
    minst_test = datasets.FashionMNIST(root=root, train=False, transform=transformer, download=False)

    #根据所处的系统环境来决定是否用额外的进程来加速读取数据，0表示不使用额外的进程
    num_workers = 0 if sys.platform.startswith('win32') else 4

    #每次读取一个样本数为batch_size的小批量数据，shuffle为True表示打乱顺序读取，即随机读取
    train_iter = DataLoader(minst_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(minst_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


#定义模型
def net(X, num_inputs = 784):
    #因为加载的数据中图片依然是28*28的，首先应将其变换为1*784的行向量，每一个像素都代表一个特征
    Output = torch.mm(X.reshape(-1, num_inputs), w) + b

    #通过softmax运算，将输出变成合法的概率分布，即对所有类别的预测概率之和为1
    Output_exp = Output.exp()
    partition = Output_exp.sum(dim=1, keepdim=True)

    return Output_exp / partition  #广播运算


#定义损失函数
def cross_entropy(y_hat, y):
    #交叉熵损失函数
    #这里对交叉熵函数进行了简化，只考虑了对正确的类别的预测概率，如果一张图片有多个标签，则不能进行简化
    #gather函数，根据索引index来读取y_hat中的数据，dim=1表示按行读取
    return -torch.gather(y_hat, dim=1, index=y.reshape(y_hat.shape[0], -1)).log()


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
    test_acc_sum = 0
    n = 0
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
    train_iter, test_iter = load_data_fashion_minst(root="~/dataset/", batch_size=batch_size)
    #设置特征和输出的个数
    num_inputs = 784
    num_outputs = 10
    # 初始化模型参数
    w = torch.normal(mean=torch.zeros(num_inputs, num_outputs), std=0.01)
    b = torch.zeros(num_outputs)
    #初始化参数的梯度，使在后面的训练过程中能够利用梯度来更新参数
    w.requires_grad_()
    b.requires_grad_()
    #设置迭代次数
    num_epoch = 5
    #设置学习率
    lr = 0.1
    #训练
    train(net, train_iter, loss=cross_entropy, num_epoch=num_epoch, batch_size=batch_size, params=[w, b], lr=lr)
    #耗时
    print("time span %f" % (time.time()-start))
    #测试
    test(net, test_iter)