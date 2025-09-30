---
title: 10.6 动量法
date: 2024-8-7 14:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---
#  
<!--more-->
# 6 动量法

- 本节将探讨更有效的优化算法

## 6.1 基础

### 6.1.1 泄露平均值

- 小批量随机梯度下降通过如下方式计算:
    $$ \begin{aligned} \boldsymbol{g}_ {t, t-1 } & = \partial_{\boldsymbol{w}} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} f(\boldsymbol{x}_i, \boldsymbol{w} _ { t- 1})\\\\ & = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{h}_{i, t-1} \end{aligned} $$

    - 其中$\mathbf{h}_{i, t-1} = \partial_{\boldsymbol{w}} f(\boldsymbol{x}_i, \boldsymbol{w} _ { t- 1})$是样本i的随机梯度下降 (使用时间t-1时更新的权重)

- 它也有很好的副作用，即平均梯度减小了方差。如果我们能够从方差减少的影响中受益，甚至超过小批量上的梯度平均值，会有好处. 可以用泄漏平均值（leakyaverage）取代梯度计算：
$$ \mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1} $$

- 其中, $\beta \in [0, 1)$. 将瞬时梯度替换为多个“过去”梯度的平均值。v被称为动量（momentum），它累加了过去的梯度:
    $$ \begin{aligned} \mathbf{v}_t & = \beta \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1} \\\\ & = ... \\\\ & = \sum_{\tau=0}^{t-1} \beta^{ \tau} \mathbf{g}_{t-\tau, t-\tau-1} \end{aligned} $$

- 其中，较大的$\beta$相当于长期平均值，而较小的$\beta$相对于梯度法只是略有修正。新的梯度替换不再指向特定实例下降最陡的方向，而是指向过去梯度的加权平均值的方向。这使我们能够实现对单批量计算平均值的大部分好处，而不产生实际计算其梯度的代价。

- 上述推理构成了“加速”梯度方法的基础，例如具有动量的梯度。在优化问题条件不佳的情况下（例如，有些方向的进展比其他方向慢得多，类似狭窄的峡谷），“加速”梯度还额外享受更有效的好处。此外，它们允许我们对随后的梯度计算平均值，以获得更稳定的下降方向。

### 6.1.2 条件不佳的问题

- 定义函数: $f(\boldsymbol{x}) = 0.1x_1^2 + 2x_2^2$
    - f在(0,0)有最小值, 该函数在x1的方向上非常平坦, 在这个函数上执行梯度下降:


```python
%matplotlib inline
import torch
from d2l import torch as d2l

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2* x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

    epoch 20, x1: -0.943467, x2: -0.000073
    


    
![svg](img/deeplearning/code/pytorch/10_optimize/6_momentum_files/6_momentum_1_1.svg)
    


- 由上图可知: x2方向的梯度比水平x1方向的梯度大得多，变化也快得多。因此，我们陷入两难：如果选择较小的学习率，我们会确保解不会在x2方向发散，但要承受在x1方向的缓慢收敛。相反，如果学习率较高，我们在x1方向上进展很快，但在x2方向将会发散。可以看到上图x1方向进展缓慢. 

- 将学习率从0.4略微提高到0.6，x1方向上的收敛有所改善，但整体来看解的质量更差了。


```python
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

    epoch 20, x1: -0.387814, x2: -1673.365109
    


    
![svg](img/deeplearning/code/pytorch/10_optimize/6_momentum_files/6_momentum_3_1.svg)
    


## 6.2 动量法

- 观察上面的优化轨迹，我们可能会直觉到计算过去的平均梯度效果会很好。毕竟，在x1方向上，这将聚合非常对齐的梯度，从而增加我们在每一步中覆盖的距离。相反，在梯度振荡的x2方向，由于相互抵消了对方的振荡，聚合梯度将减小步长大小:

    $$ \begin{aligned} & \mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1} \\\\ & \mathbf{ x } _ { t } = \mathbf{ x } _ { t-1 } - \eta \mathbf{v}_t \end{aligned} $$

    - 当$\beta = 0$时，动量法等价于小批量随机梯度下降。


```python
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

    epoch 20, x1: 0.007188, x2: 0.002553
    


    
![svg](img/deeplearning/code/pytorch/10_optimize/6_momentum_files/6_momentum_5_1.svg)
    


- 尽管学习率与我们以前使用的相同，动量法仍然很好地收敛了。
- 当降低动量参数时, 将其减半至β=0.25会导致一条几乎没有收敛的轨迹。尽管如此，它比没有动量时解将会发散要好得多。


```python
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

    epoch 20, x1: -0.126340, x2: -0.186632
    


    
![svg](img/deeplearning/code/pytorch/10_optimize/6_momentum_files/6_momentum_7_1.svg)
    


- 可以将动量法与随机梯度下降，特别是小批量随机梯度下降结合起来。唯一的变化是，在这种情况下，我们将梯度$g_ {t,t−1}$替换为$g_ t$。为了方便起见，我们在时间t=0初始化$v_ 0$=0。

### 6.2.1 有效样本权重

- 对于$\mathbf { v }_ t = \sum_{\tau=0}^{t-1} \beta^{ \tau} \mathbf{g}_{t-\tau, t-\tau-1}$, 极限条件下, $\sum_{\tau=0}^{ \infty } \beta^{ \tau} = \frac{1}{1-\beta}$. 换句话说，不同于在梯度下降或者随机梯度下降中取步长$\eta$，我们选取步长$\frac{ \eta }{ 1- \beta}$，同时处理潜在表现可能会更好的下降方向。这是集两种好处于一身的做法。为了说明β的不同选择的权重效果如何，请参考下面的图表。


```python
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = torch.arange(40).detach().numpy()
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```


    
![svg](img/deeplearning/code/pytorch/10_optimize/6_momentum_files/6_momentum_9_0.svg)
    


## 6.2 实验

### 6.2.1 从零开始

- 相比于小批量随机梯度下降，动量方法需要维护一组辅助变量，即速度。它与梯度以及优化问题的变量具有相同的形状。在下面的实现中，我们称这些变量为states。


```python
def init_momentum_states(feature_dim):
    v_w = torch.zeros((feature_dim, 1))
    v_b = torch.zeros(1)
    return (v_w, v_b)
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter, feature_dim, num_epochs)
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
    
```

    loss: 0.245, 0.008 sec/epoch
    


    
![svg](img/deeplearning/code/pytorch/10_optimize/6_momentum_files/6_momentum_11_1.svg)
    


- momentum增加到0.9时，它相当于有效样本数量增加到$\frac{1}{1-0.9} = 10$。我们减小学习率$\eta$=0.01:


```python
train_momentum(0.01, 0.9)
```

    loss: 0.250, 0.009 sec/epoch
    


    
![svg](img/deeplearning/code/pytorch/10_optimize/6_momentum_files/6_momentum_13_1.svg)
    


- 降低学习率进一步解决了任何非平滑优化问题的困难，将其设置为0.005会产生良好的收敛性能。


```python
train_momentum(0.005, 0.9)
```

    loss: 0.244, 0.008 sec/epoch
    


    
![svg](img/deeplearning/code/pytorch/10_optimize/6_momentum_files/6_momentum_15_1.svg)
    


### 6.2.2 简介实现


```python
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

    loss: 0.254, 0.008 sec/epoch
    


    
![svg](img/deeplearning/code/pytorch/10_optimize/6_momentum_files/6_momentum_17_1.svg)
    


## 6.3 理论分析

### 6.3.1 二次凸函数

- 考虑一个普通的二次函数:
    $$ h (  \mathbf{ x } ) = \frac{1}{2} \mathbf{ x } ^ { T } \mathbf{ Q } \mathbf{ x } + \mathbf{ c } \mathbf{ x } ^ { T } + b $$

- 对于正定矩阵$\mathbf{ Q }$>0, 有最小化器$ \mathbf{ x } ^ { * } = - \mathbf{ Q } ^ { -1 } \mathbf{ c }$，最小值为$b - \frac{1}{2} \mathbf{ c } ^ { T } \mathbf{ Q } ^ { -1 } \mathbf{ c } $, 因此h可重写为:
    $$ h (  \mathbf{ x } ) = \frac{1}{2} ( \mathbf{ x } - \mathbf{ Q } ^ { -1 } \mathbf{ c } ) ^ { T } \mathbf{ Q } ( \mathbf{ x } - \mathbf{ Q } ^ { -1 } \mathbf{ c } ) + b - \frac{1}{2} \mathbf{ c } ^ { T } \mathbf{ Q } ^ { -1 } \mathbf{ c } $$

    - 梯度由$\partial_{\mathbf{ x }} f (  \mathbf{ x } ) = \mathbf{ Q } ( \mathbf{ x } - \mathbf{ Q } ^ { -1 } \mathbf{ c } )$给出也就是说，它是由x和最小化器之间的距离乘以Q所得出的。因此，动量法还是$\mathbf{ Q }(\mathbf{ x }_ t− \mathbf{ Q }^ {−1} \mathbf{ c })$的线性组合。

- 由于$\mathbf{ Q }$正定, 因此$\mathbf{ Q } = \mathbf{ O }^ { T } \mathbf{ \Lambda } \mathbf{ O }$，其中$\mathbf{ O }$是正交矩阵，$\mathbf{ \Lambda }$是正特征值的对角矩阵. 因此定义: $\mathbf{ z } = \mathbf{ O } ( \mathbf{ x } - \mathbf{ Q } ^ { -1 } \mathbf{ c })$，则有:
    $$h ( \mathbf { z } ) = \frac{1}{2} \mathbf{ z } ^ { T } \mathbf{ \Lambda } \mathbf{ z } + b'$$

    - 以z表示的梯度下降:
        $$\mathbf{ z }_t = \mathbf{ z }_{t-1} - \mathbf{ \Lambda } \mathbf{ z }_{t-1} = (1- \mathbf{ \Lambda }) \mathbf{ z }_{t-1}$$

    - 上式表明梯度下降在不同的特征空间之间不会混合。也就是说，如果用Q的特征系统来表示，优化问题是以逐坐标顺序的方式进行的。这在动量法中也适用。
        $$ \begin{aligned} \mathbf{ z }_t & = \beta \mathbf{ v }_{t-1} + \mathbf{ \Lambda } \mathbf{ z }_{t-1} \\\\ \mathbf{ z }_ t & = \mathbf{ z }_ { t-1 } - \eta ( \beta \mathbf{ v }_{t-1} + \mathbf{ \Lambda } \mathbf{ z }_{t-1} ) \\\\ & = (1- \eta \mathbf{ \Lambda }) \mathbf{ z }_{t-1} - \eta \beta \mathbf{ v }_{t-1} \end{aligned} $$

- 上述过程说明: 带有和带有不凸二次函数动量的梯度下降，可以分解为朝二次矩阵特征向量方向坐标顺序的优化。

### 6.3.2 标量函数

- 鉴于上述结果, 我们分析函数: $f(x) = \frac{ \lambda }{2} x^2$, 梯度下降:
    $$ x_{ t+ 1 } = x_t - \eta ( \lambda x_t ) = (1- \eta \lambda) x_t $$

    - 当$|\eta \lambda| < 1$时，这种优化以指数速度收敛，因为在t步之后$x_t = (1- \eta \lambda)^t x_0$。这显示了在我们将学习率η提高到ηλ=1之前，收敛率最初是如何提高的。超过该数值之后，梯度开始发散，对于ηλ>2而言，优化问题将会发散。


```python
lambdas = [0.1, 1, 10, 19]
eta = 0.1
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = torch.arange(20).detach().numpy()
    d2l.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```


    
![svg](img/deeplearning/code/pytorch/10_optimize/6_momentum_files/6_momentum_19_0.svg)
    


- 为了分析动量的收敛情况，我们首先用两个标量重写更新方程：一个用于x，另一个用于动量v。
    $$\begin{bmatrix}v _ { t+1 } \\\\ x _ { t+1 }\end{bmatrix}= \begin{bmatrix} \beta & \lambda \\\\ -\eta\beta &(1-\eta\lambda)\end{bmatrix} \begin{bmatrix} v _ t \\\\ x_ t\end{bmatrix}= \mathbf{R}(\beta,\eta,\lambda)\begin{bmatrix}v_t \\x_t \end{bmatrix}$$

- 在t步之后，最初的值$[v_0,x_0]$变为$\mathbf{ R }(\beta ,\eta ,\lambda)^ t[v_ 0,x_ 0]$。因此，收敛速度是由R的特征值决定的。简而言之，当0<ηλ<2+2β时动量收敛。与梯度下降的0<ηλ<2相比，这是更大范围的可行参数。另外，一般而言较大值的β是可取的。
