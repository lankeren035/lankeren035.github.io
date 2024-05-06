---
title: 4.1 层和块
date: 2024-4-18 14:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---
#  
<!--more-->
# 1 层和块
- 单个神经网络：
    - 一个输入
    - 标量输出
    - 一组相关参数，这些参数可以通过学习而优化
- 层：
    - 一组输入
    - 一组输出
    - 一组可调参数
- 从编程的角度看，块由类来表示。通常需要定义一个将输入转换成输出的forward函数，并且必须存储任何必须的参数。
- 定义一个网络：256个单元和ReLU的全连接隐藏层，10个隐藏单元且不带激活函数的全连接输出层
    - nn.Sequential是一种特殊的module，表示一个块，维护了一个由module组成的有序列表
    - net(x)相当于net.__call__(x)


```python
import torch
from torch import nn
from torch.nn import functional as F

net= nn.Sequential( #一种特殊的module，表示一个块，维护了一个由module组成的有序列表
    nn.Linear(20,256),
    nn.ReLU(), 
    nn.Linear(256,10))

X = torch.rand(2,20)
print(net(X))
```

    tensor([[ 0.0302, -0.1207, -0.0915,  0.0522, -0.3685,  0.0474,  0.0665, -0.0055,
             -0.1445, -0.1954],
            [ 0.0275, -0.1345,  0.0009,  0.0987, -0.4689,  0.0405,  0.0012, -0.0243,
             -0.2954, -0.1414]], grad_fn=<AddmmBackward0>)
    

## 1.1 自定义块
- 每个块必须提供的功能：
    - 数据输入forward函数得到输出
    - 计算输出关于输入的梯度，通过backward函数
    - 存储和访问前向传播计算所需要的参数
    - 初始化模型参数


```python
class MLP(nn.Module):
    # 用模型参数声明层。声明两个全连接层
    def __init__(self):
        super().__init__() #Module的构造函数进行必要的初始化
        self.hidden = nn.Linear(20,256) #隐藏层
        self.out = nn.Linear(256,10) #输出层

    # 前行传播，如何根据输入x返回所需的模型输出
    def forward(self,X):
        return self.out(F.relu(self.hidden(X)))
    
net = MLP()
print(net(X))
```

    tensor([[ 0.2163,  0.0644, -0.1128, -0.2947,  0.0708,  0.0907, -0.0680, -0.0381,
             -0.0843,  0.0921],
            [ 0.2196,  0.0087,  0.0257, -0.1403, -0.0191,  0.0435, -0.1980,  0.0350,
              0.0158,  0.0848]], grad_fn=<AddmmBackward0>)
    

## 1.2 顺序块
- 构建自己简化的Sequential类需要
    - 将block逐个追加到列表中
    - forward函数中，将输入按顺序传递


```python
class MySequential(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()
        for idx, module in enumerate(args):
            # module是Module子类的一个实例
            # _modules中，_module的类型是OrderedDict
            #为啥每个Module都有一个_modules属性，为啥不用python列表？
            # _modules优点：在模块的参数初始化过程中，系统知道在_modules字典中查找需要初始化参数的子块。
            self._modules[str(idx)] = module
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X
    
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256,10))
print(net(X))
```

    tensor([[-0.1585, -0.0420,  0.3813, -0.3592, -0.0889,  0.0362, -0.0543, -0.0557,
             -0.2022,  0.0183],
            [-0.0438, -0.1248,  0.5774, -0.3087, -0.0576, -0.0479,  0.0954, -0.2362,
             -0.2333, -0.1394]], grad_fn=<AddmmBackward0>)
    

## 1.3 在前向传播函数中执行代码
- 有时我们希望合并既不是上一层的结果也不是可更新参数的项，成为常数参数。


```python
class FixedHiddenMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 不计算梯度的随机权重参数，因此其在训练期间保持不变
        self.rand_weight = torch.rand((20,20), requires_grad=False)
        self.linear = nn.Linear(20,20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。相当于两个全连接成共享参数
        X = self.linear(X)
        # 控制流：
        while X.abs().sum() > 1: #l1范数
            X/=2
        return X.sum()
net = FixedHiddenMLP()
print(net(X))
```

    tensor(-0.2985, grad_fn=<SumBackward0>)
    

## 1.4 效率
我们在一个高性能的深度学习库中进行了大量的字典查找、代码执行和许多其他的Python代码。Python的问题全局解释器锁是众所周知的。在深度学习环境中，我们担心速度极快的GPU可能要等到CPU运行Python代码后才能运行另一个作业。
