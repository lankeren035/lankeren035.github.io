---
title: 4.2 参数管理
date: 2024-4-19 14:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---
#
<!--more-->
# 2 参数管理
- 访问参数，用于调试、诊断和可视化
- 参数初始化
- 在不同模型组件间共享参数


```python
'''单隐藏层的多层感知机'''
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,1))
X = torch.rand(size=(2,4))
print(net(X))
```

    tensor([[0.2326],
            [0.2463]], grad_fn=<AddmmBackward0>)
    

## 2.1 参数访问
- 对于Sequential类模型，通过索引访问参数


```python
print(net[2].state_dict())
#输出的这个层包含两个参数，分别是权重和偏置。
```

    OrderedDict([('weight', tensor([[ 0.0830, -0.2212, -0.3532,  0.2318,  0.2688, -0.1233, -0.2256, -0.1163]])), ('bias', tensor([0.1769]))])
    

### 2.1.1 目标参数
- 提取第三层的偏置


```python
param = net[2].bias

print(param)
print(type(param)) #类型
print(param.data) #数据
```

    Parameter containing:
    tensor([0.1769], requires_grad=True)
    <class 'torch.nn.parameter.Parameter'>
    tensor([0.1769])
    

- 参数是复合对象，包含值、梯度和额外信息


```python
# 访问每个参数的梯度
print(net[2].weight.grad == None)
```

    True
    

- 访问所有参数


```python
# 访问第一个全连接层的参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
# 访问所有层
print(*[(name, param.shape) for name, param in net.named_parameters()])
# 访问一个参数
print(net.state_dict()['2.bias'].data)
```

    ('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
    ('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
    tensor([0.1769])
    

- 从嵌套块收集参数


```python
def block1():
    return nn.Sequential(
        nn.Linear(4,8),nn.ReLU(),
        nn.Linear(8,4), nn.ReLU()
    )
def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}',block1()) #增加一个模块
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4,1))
print(rgnet(X))
# 输出网络
print(rgnet)
# 访问第一个主要的模块中第二个子模块第一层的偏置
print(rgnet[0][1][0].bias.data)
```

    tensor([[-0.2233],
            [-0.2233]], grad_fn=<AddmmBackward0>)
    Sequential(
      (0): Sequential(
        (block0): Sequential(
          (0): Linear(in_features=4, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
        )
        (block1): Sequential(
          (0): Linear(in_features=4, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
        )
        (block2): Sequential(
          (0): Linear(in_features=4, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
        )
        (block3): Sequential(
          (0): Linear(in_features=4, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
        )
      )
      (1): Linear(in_features=4, out_features=1, bias=True)
    )
    tensor([ 0.4794,  0.4585,  0.0098,  0.1719, -0.2016, -0.3174, -0.4974,  0.4953])
    

## 2.2 参数初始化
- 默认情况下，pytorch会根据一个均匀地初始化权重和偏置
- pytorch的nn.init模块提供了多种初始化方法
### 2.2.1 内置初始化
- 权重初始化标准差为0.01的高斯随机变量，偏置设置为0



```python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean = 0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])
```

    tensor([ 0.0058, -0.0130, -0.0114,  0.0044]) tensor(0.)
    

- 初始化为常熟


```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print(net[0].weight.data[0],net[0].bias.data[0])
```

    tensor([1., 1., 1., 1.]) tensor(0.)
    

- 不同块使用不同初始化方法：用Xavier初始化方法初始化第一层，第三层初始化为常熟42


```python
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)
net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

    tensor([ 0.2578, -0.6072, -0.2665,  0.4973])
    tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])
    

### 2.2.2 自定义初始化
$$\omega =\left\{ \begin{matrix}U(5,10) &可能性\frac{1}{4} \\
0 &可能性\frac{1}{2}\\
U(-10,-5) &可能性\frac{1}{4} \\
\end{matrix}\right.$$


```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10,10) #参数-> [-10,10]
        m.weight.data *= m.weight.data.abs() >= 5 # 参数abs>=5的概率=1/2 ， 1/2的概率=0，1/2的概率在两个分布
net.apply(my_init)
print(net[0].weight[:2])
# 直接设置参数
net[0].weight.data[:] += 1
net[0].weight.data[0,0] = 42
print(net[0].weight.data[0])
```

    Init weight torch.Size([8, 4])
    Init weight torch.Size([1, 8])
    tensor([[ 8.4257,  0.0000, -7.9400, -0.0000],
            [ 0.0000,  8.8472,  5.5845, -0.0000]], grad_fn=<SliceBackward0>)
    tensor([42.0000,  1.0000, -6.9400,  1.0000])
    

## 2.3 参数绑定
- 有时我们希望在多个层之间共享参数


```python
# 定义一个共享层
shared = nn.Linear(8,8)
net = nn.Sequential(nn.Linear(4,8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8,1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0,0] = 100
# 确保它们是同一个对象，而不只是具有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
```

    tensor([True, True, True, True, True, True, True, True])
    tensor([True, True, True, True, True, True, True, True])
    

- 当参数绑定时，梯度会发生什么情况：由于模型参数包含梯度，因此在反向传播期间第二个隐藏层（net[2]）和第三个隐藏层（net[5]）的梯度会加在一起
