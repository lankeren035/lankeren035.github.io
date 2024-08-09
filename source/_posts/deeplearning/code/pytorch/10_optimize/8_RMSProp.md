---
title: 10.8 RMSProp算法
date: 2024-8-8 13:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---
#  
<!--more-->
# 8 RMSProp算法

- 上一节的关键问题之一，是学习率按预定时间表$O(t^{-\frac12})$显著降低。虽然这通常适用于凸问题，但对于深度学习中遇到的非凸问题，可能并不理想。但是，作为一个预处理器，Adagrad算法按坐标顺序的适应性是非常可取的。

- 问题在于，Adagrad算法将梯度$g_t$的平方累加成状态矢量$s_t =s_{t−1}+g^2_t$。因此，由于缺乏规范化，没有约束力，$s_t$持续增长，几乎上是在算法收敛时呈线性递增。
    - 解决此问题的一种方法是使用$\frac {s_t}{t}$对$g_t$的合理分布来说，它将收敛。遗憾的是，限制行为生效可能需要很长时间，因为该流程记住了值的完整轨迹。
    - 另一种方法是按动量法中的方式使用泄漏平均值，即$\mathbf s_t \leftarrow \gamma \mathbf s_{t-1} + (1-\gamma) \mathbf g_t^2$，其中$\gamma$>0. 保持所有其它部分不变就产生了RMSProp算法。

## 8.1 算法

$$\mathbf s_t \leftarrow \gamma \mathbf s_{t-1} + (1-\gamma) \mathbf g_t^2$$
$$\mathbf x_t \leftarrow \mathbf x_{t-1} - \frac{\eta}{\sqrt{\mathbf s_t + \epsilon}} \odot \mathbf g_t$$

- 我们现在可以自由控制学习率η，而不考虑基于每个坐标应用的缩放。就泄漏平均值而言，我们可以采用与之前在动量法中适用的相同推理。扩展$\mathbf s_t $定义可获得:
    $$\begin{aligned} \mathbf s_t &= (1-\gamma) \mathbf g_t^2 + \gamma \mathbf s_{t-1} \\& = (1-\gamma) \mathbf g_t^2 + \gamma (1-\gamma) \mathbf g_{t-1}^2 + \gamma^2 \mathbf s_{t-2} \\& = \sum_{i=0}^{t-1} \gamma^{t-i} (1-\gamma) \mathbf g_i^2 \end{aligned}$$

    - 我们使用$1+\gamma+\gamma^2+\cdots = \frac{1}{1-\gamma}$。因此，权重总和标准化为1且观测值的半衰期为$\gamma^ {-1}$。让我们图像化各种数值的γ在过去40个时间步长的权重。


```python
import math
import torch
from d2l import torch as d2l

d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = torch.arange(40).detach().numpy()
    d2l.plt.plot(x, (1 - gamma) * gamma**x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('Time')
```




    Text(0.5, 0, 'Time')




    
![svg](img/deeplearning/code/pytorch/10_optimize/8_RMSProp_files/8_RMSProp_1_1.svg)
    


## 8.2 从零开始实现
- 对$f ( \mathbf x ) = 0.1x_1^2 + 2x_2^2$使用RMSProp算法。当我们使用学习率为0.4的Adagrad算法时，变量在算法的后期阶段移动非常缓慢，因为学习率衰减太快。RMSProp算法中不会发生这种情况，因为η是单独控制的。


```python
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1**2
    s2 = gamma * s2 + (1 - gamma) * g2**2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1**2 + 2 * x2**2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

    epoch 20, x1: -0.010599, x2: 0.000000
    

    c:\Users\admin\miniconda3\envs\d2l\lib\site-packages\torch\functional.py:478: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\native\TensorShape.cpp:2895.)
      return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
    


    
![svg](img/deeplearning/code/pytorch/10_optimize/8_RMSProp_files/8_RMSProp_3_2.svg)
    


### 8.2.1 在深度网络中使用RMSProp算法


```python
def init_rmsprop_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))
    s_b = torch.zeros(1)
    return (s_w, s_b)
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()

#初始学习率设置为0.01，加权项γ设置为0.9。也就是说，s累加了过去的1/(1−γ)=10次平方梯度观测值的平均值。
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

    loss: 0.244, 0.009 sec/epoch
    


    
![svg](img/deeplearning/code/pytorch/10_optimize/8_RMSProp_files/8_RMSProp_5_1.svg)
    


## 8.3 简洁实现


```python
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9}, data_iter)
```

    loss: 0.242, 0.009 sec/epoch
    


    
![svg](img/deeplearning/code/pytorch/10_optimize/8_RMSProp_files/8_RMSProp_7_1.svg)
    

