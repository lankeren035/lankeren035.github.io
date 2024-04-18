---
title: 3.3 多层感知机简洁实现
date: 2024-2-5 14:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---
#
<!--more-->
# 3 多层感知机简洁实现


```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"]= 'True'
import torch
from torch import nn
from d2l import torch as d2l

#1 模型
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28,256),
    nn.ReLU(),
    nn.Linear(256,10)
)
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
net.apply(init_weights)

#2 参数
batch_size, lr, num_epochs= 256, 0.1, 10

#3 损失
loss= nn.CrossEntropyLoss(reduction='none')

#4 优化器
trainer= torch.optim.SGD(net.parameters(),lr=lr)

#5 训练
train_iter, test_iter= d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```


  ![svg](D:/blog/themes/yilia/source/img/deeplearning/code/pytorch/3_mlp/3_mlp_realize_simple_files/3_mlp_realize_simple_1_0.svg)
![](img/deeplearning/code/pytorch/3_mlp/3_mlp_realize_simple_files/3_mlp_realize_simple_1_0.svg)
    

