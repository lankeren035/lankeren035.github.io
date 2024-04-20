---
title: 4.4 自定义层
date: 2024-4-20 14:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---
#
<!--more-->
# 4 自定义层
## 4.1 不带参数的层
- 创建一个CenteredLayer，接受一个输入，输出内容是输入减去输入的均值


```python
import torch
import torch.nn.functional as F
from torch import nn
class CenteredLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, X):
        return X-X.mean()
layer = CenteredLayer()
layer(torch.FloatTensor([1,2,3,4,5]))
```




    tensor([-2., -1.,  0.,  1.,  2.])



- 将层作为组件合并到更复杂的模型中


```python
net = nn.Sequential(nn.Linear(8,128), CenteredLayer())

'''向网络发送随机数据，检查均值是否为0'''
Y = net(torch.rand(4,8))
print(Y.mean())
```

    tensor(2.0955e-09, grad_fn=<MeanBackward0>)
    

## 4.2 带参数的层
- 使用内置函数来创建参数，这些函数可以：管理访问、初始化、共享、保存、加载参数
- 自定义全连接层


```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units,units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
linear = MyLinear(5,3)
print(linear.weight) #访问模型参数
print(linear(torch.rand(2,5))) #前向传播
```

    Parameter containing:
    tensor([[-1.2740, -0.5589, -0.0577],
            [ 0.2042, -2.4343, -1.4043],
            [ 2.3896,  0.5617, -0.1255],
            [-0.7957, -0.7547, -0.5688],
            [-1.3496,  0.6484,  0.9639]], requires_grad=True)
    tensor([[0., 0., 0.],
            [0., 0., 0.]])
    

- 用自定义层构建模型


```python
net = nn.Sequential(MyLinear(64,8), MyLinear(8,1))
print(net(torch.rand(2,64)))
```

    tensor([[ 5.7095],
            [10.4373]])
    
