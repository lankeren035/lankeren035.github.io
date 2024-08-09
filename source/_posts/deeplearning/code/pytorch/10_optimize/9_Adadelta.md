---
title: 10.9 Adadelta算法
date: 2024-8-8 14:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---
#  
<!--more-->
# 9 Adadelta算法

-  Adadelta是AdaGrad的另一种变体主要区别在于前者减少了学习率适应坐标的数量。此外，广义上Adadelta被称为没有学习率，因为它使用变化量本身作为未来变化的校准。

## 9.1 Adadelta算法
- Adadelta使用两个状态变量，$s_t$用于存储梯度二阶导数的泄露平均值，$\Delta x_t$用于存储模型本身中参数变化二阶导数的泄露平均值。
    $$\mathbf{s}_t = \rho \mathbf{s}_{t-1} + (1-\rho) \mathbf{g}^2_t $$
    $$\mathbf{x}_t = \mathbf{x}_{t-1} - \mathbf{ g '}_t$$

- 其中与RMSProp的区别在于使用重新缩放的梯度$g′_t$执行更新:
    $$\mathbf{g}'_t = \frac{\sqrt{\Delta \mathbf{x}_{t-1} + \epsilon}}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t$$
    $$\Delta \mathbf{x}_t = \rho \Delta \mathbf{x}_{t-1} + (1-\rho) \mathbf{g}'^2_t$$

    - 其中$\Delta x_t$是重新缩放梯度的平方$\mathbf{g}'^2_t$的泄漏平均值，$\Delta x_0$初始化为0。

## 9.2 从零实现


```python
%matplotlib inline
import torch
from d2l import torch as d2l

def init_adadelta_states(feature_dim):
    s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    delta_w, delta_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = ((torch.sqrt(delta + eps) / torch.sqrt(s + eps)) *
                 p.grad)
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

    loss: 0.245, 0.011 sec/epoch
    


    
![svg](img/deeplearning/code/pytorch/10_optimize/9_Adadelta_files/9_Adadelta_1_1.svg)
    


## 9.3 简洁实现


```python
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

    loss: 0.243, 0.009 sec/epoch
    


    
![svg](img/deeplearning/code/pytorch/10_optimize/9_Adadelta_files/9_Adadelta_3_1.svg)
    

