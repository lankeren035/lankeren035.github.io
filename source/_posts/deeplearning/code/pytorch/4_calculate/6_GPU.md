---
title: 4.6 GPU
date: 2024-4-20 17:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---
#  
<!--more-->
# 6 GPU
- 查看显卡信息


```python
!nvidia-smi
```

- pytorch中，每个数组都有一个设备，称为：环境。默认情况下，所有变量和相关的计算都分配给CPU。
## 6.1 计算设备



```python
import torch
from torch import nn

torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')
#查询可用GPU数量
print(torch.cuda.device_count())
```

    0
    

- 选择GPU或CPU


```python
def try_gpu(i=0): #@save
    '''如果存在，返回GPU(i),否则cpu()'''
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
def try_all_gpus(): #@save
    '''返回所有可用GPU，无则cpu'''
    devices = [
        torch.device(f'cuda:{i}')
        for i in range(torch.cuda.device_count())
               ]
    return devices if devices else[torch.device('cpu')]

print(try_gpu())
print(try_gpu(10))
print(try_all_gpus())
```

    cpu
    cpu
    [device(type='cpu')]
    

## 6.2 张量与GPU
- 查看张量所在设备


```python
x = torch.tensor([1,2,3])
print(x.device)
```

    cpu
    

### 6.2.1 存储在GPU上


```python
X = torch.ones(2,3,device=try_gpu())
print(X)
```

    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    

### 6.2.2 复制
- 两个变量处于不同的GPU上，不能直接计算，需要将其中一个变量复制到另一个GPU上。
- 如果z在GPU0上，z.cuda(0)不会复制


```python
Z = X.cuda(0) #等价于X.to(torch.device('cuda:0'))
```

### 6.2.3 旁注
- 多GPU运行时，数据传输比计算慢得多
- 一次执行几个操作比代码中散步的多个单操作快得多
- 打印张量或将张量转换为numpy格式时，如果数据不在内存中，框架会先将其复制到内存中，这会导致额外的传输开销。
## 6.3 神经网络与GPU
- 神经网络模型也可以指定设备


```python
net = nn.Sequential(nn.Linear(3,1))
net = net.to(device=try_gpu())
```
