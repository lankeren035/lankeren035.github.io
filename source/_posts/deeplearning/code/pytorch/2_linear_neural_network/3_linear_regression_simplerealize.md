---
title: 2.3 线性回归简洁实现
date: 2024-2-3 14:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---
#  
<!--more-->
# 3 线性回归简洁实现


```python
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

#1 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000) #生成方式为y = Xw + b + e，e为噪声默认为服从N(0,1)的正态分布


#2 读取数据集
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器。"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
batch_size = 10
data_iter = load_array((features, labels), batch_size)
print(next(iter(data_iter)))

#3 定义模型
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))

#4 初始化模型参数
net[0].weight.data.normal_(0, 0.01) #将第一层的权重初始化为均值为0，标准差为0.01的正态分布
net[0].bias.data.fill_(0) #将偏置初始化为0

#5 定义损失函数
loss = nn.MSELoss() #均方误差损失函数

#6 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

    [tensor([[ 0.9365, -0.9840],
            [ 0.9369,  1.5472],
            [ 0.1462,  0.3326],
            [-0.1900,  0.8358],
            [ 1.1384, -1.4987],
            [-0.4850, -0.8085],
            [-2.1479, -0.8616],
            [ 0.9232,  0.0479],
            [ 0.3618, -0.7924],
            [-1.2253,  0.1714]]), tensor([[ 9.4372],
            [ 0.8145],
            [ 3.3749],
            [ 0.9582],
            [11.5770],
            [ 5.9775],
            [ 2.8450],
            [ 5.8905],
            [ 7.6133],
            [ 1.1836]])]
    

## 2.7 训练
- 重复以下训练，知道完成：
    - net(x)生成预测并计算损失l（正向传播）
    - backward()计算梯度（反向传播）
    - 优化器更新模型参数


```python
#1 超参数
num_epochs = 3

#2 训练
for epoch in range(num_epochs):
    for X,y in data_iter:
        l = loss(net(X),y) #l是小批量X和y的损失
        trainer.zero_grad()
        l.backward()
        trainer.step() #更新模型参数
    l = loss(net(features),labels) #整个数据集的损失
    print(f'epoch {epoch + 1}, loss {l:f}')

#3 检验
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：',true_b-b)
```

    epoch 1, loss 0.000270
    epoch 2, loss 0.000103
    epoch 3, loss 0.000103
    w的估计误差： tensor([-0.0003, -0.0005])
    b的估计误差： tensor([0.0007])
    
