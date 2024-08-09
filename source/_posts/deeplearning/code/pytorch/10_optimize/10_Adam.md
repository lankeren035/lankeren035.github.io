---
title: 10.10 Adam算法
date: 2024-8-8 15:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---
#  
<!--more-->
# 10 Adam算法

- 回顾:
    - 10.4: SGD在解决优化问题时比GD更有效。
    - 10.5: 在一个小批量中使用更大的观测值集，可以通过向量化提供额外效率。这是高效的多机、多GPU和整体并行处理的关键。
    - 10.6: 添加了一种机制，用于汇总过去梯度的历史以加速收敛。
    - 10.7: 通过对每个坐标缩放来实现高效计算的预处理器。
    - 10.8: 通过学习率的调整来分离每个坐标的缩放。

- Adam算法将所有这些技术汇总到一个高效的学习算法中。

- 它并非没有问题，有时Adam算法可能由于方差控制不良而发散。可用Yogi的热补丁来解决这些问题

## 10.1 算法

- Adam算法使用指数加权移动平均值来估算梯度的动量和二次矩，即它使用状态变量
    $$ \mathbf{ v }_t \leftarrow \beta_1 \mathbf{ v }_{t-1} + (1-\beta_1) \mathbf{ g }_t $$
    $$ \mathbf{ s }_t \leftarrow \beta_2 \mathbf{ s }_{t-1} + (1-\beta_2) \mathbf{ g }_t^2 $$

    - 这里β1和β2是非负加权参数。常将它们设置为β1 =0.9和β2 = 0.999。也就是说，方差估计的移动远远慢于动量估计的移动。注意，如果我们初始化v0 =s0 =0，就会获得一个相当大的初始偏差。我们可以通过使用$\sum^ t_ {i=1} \beta^{i} = \frac{1-\beta^t}{1-\beta}$来纠正这个偏差。相应地，标准化状态变量由下式获得:
        $$ \hat{ \mathbf{ v } }_t = \frac{ \mathbf{ v }_t }{ 1 - \beta_1^t } $$
        $$ \hat{ \mathbf{ s } }_t = \frac{ \mathbf{ s }_t }{ 1 - \beta_2^t } $$
    
    - 有了正确的估计，我们现在可以写出更新方程。首先，我们以非常类似于RMSProp算法的方式重新缩放梯度以获得:
        $$ \mathbf{ g'}_t = \frac{ \eta \hat{ \mathbf{ v } }_t }{ \sqrt{ \hat{ \mathbf{ s } }_t } + \epsilon } $$

    - 与RMSProp不同，我们的更新使用动量$\mathbf{\hat{ v }}_t$而不是梯度$\mathbf g_t$. 此外，由于使用$\frac{1}{\sqrt{\hat{\mathbf s}_t} + \epsilon}$而不是$\frac{1}{\sqrt{\mathbf s_t + \epsilon} }$进行缩放，两者会略有差异。前者在实践中效果略好一些，因此与RMSProp算法有所区分。通常，我们选择$\epsilon=10^ {−6}$，这是为了在数值稳定性和逼真度之间取得良好的平衡。

- 最后,我们更新参数:
    $$ \mathbf{ x }_t \leftarrow \mathbf{ x }_{t-1} - \mathbf{ g'}_t $$

- 回顾Adam算法，它的设计灵感很清楚：首先，动量和规模在状态变量中清晰可见，它们相当独特的定义使我们移除偏项（这可以通过稍微不同的初始化和更新条件来修正）。其次，RMSProp算法中两项的组合都非常简单。最后，明确的学习率η使我们能够控制步长来解决收敛问题。

## 10.2 实现

- 为方便起见，我们将时间步t存储在hyperparams字典中。除此之外，一切都很简单


```python
%matplotlib inline
import torch
from d2l import torch as d2l

def init_adam_states(feature_dim):
    v_w, v_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * p.grad**2
            v_bias_corr = v / (1 - beta1**hyperparams['t'])
            s_bias_corr = s / (1 - beta2**hyperparams['t'])
            p.data -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr) + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

    loss: 0.244, 0.011 sec/epoch
    


    
![svg](img/deeplearning/code/pytorch/10_optimize/10_Adam_files/10_Adam_1_1.svg)
    


- 简洁实现:


```python
trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

    loss: 0.245, 0.011 sec/epoch
    


    
![svg](img/deeplearning/code/pytorch/10_optimize/10_Adam_files/10_Adam_3_1.svg)
    


## 10.3 Yogi

- Adam算法也存在一些问题：即使在凸环境下，当$\mathbf{ s }_t$的二次矩估计值爆炸时，它可能无法收敛。重写Adam算法更新如下：
    $$ \mathbf{ s }_ t \leftarrow \mathbf{ s }_{t-1} + (1-\beta_2) (\mathbf{ g }_t^2 - \mathbf{ s }_{t-1}) $$

- 每当$\mathbf{g}^2_t$具有值很大的变量或更新很稀疏时，$\mathbf{s}_t$可能会太快地“忘记”过去的值。一个有效的解决方法是将$\mathbf{ g }_t^2 - \mathbf{ s }_{t-1} $替换为$\mathbf{ g }_t^2 \odot \text{sign}(\mathbf{ g }^2_{t} - \mathbf{ s }_{t-1})$，现在更新的规模不再取决于偏差的量:
    $$ \mathbf{ s }_ t \leftarrow \mathbf{ s }_{t-1} + (1-\beta_2) \mathbf{ g }_t^2 \odot \text{sign}(\mathbf{ g }^2_{t} - \mathbf{ s }_{t-1}) $$


```python
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(torch.square(p.grad) - s)* torch.square(p.grad)
            v_bias_corr = v / (1 - beta1**hyperparams['t'])
            s_bias_corr = s / (1 - beta2**hyperparams['t'])
            p.data -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr) + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

    loss: 0.244, 0.011 sec/epoch
    


    
![svg](img/deeplearning/code/pytorch/10_optimize/10_Adam_files/10_Adam_5_1.svg)
    

