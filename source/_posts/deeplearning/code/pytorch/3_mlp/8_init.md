---
title: 3.8 数值稳定性和模型初始化
date: 2024-3-16 14:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---
#  
<!--more-->
# 8 数值稳定性和模型初始化
- 初始化方案的选择在神经网络学习中起着举足轻重的作用，它对保持数值稳定性至关重要。

## 8.1 梯度消失和梯度爆炸
- 假设网络有L层，输入X，输出o，每一层$l$由变换$f_l$定义，该变换的参数为权重$\mathbf{W}^{(l)}$，其隐藏变量是$\mathbf{h}^{(l)}$。网络定义为：
    $$\mathbf{h}^{(l)}= f_l(\mathbf{h}^{(l-1)})$$
    - 因此：$\mathbf{o}=f_L\circ\ldots\circ f_1(\mathbf{x})$
    - $\mathbf{o}$关于任何一组参数$\mathbf{W}^{(l)}$的梯度：
        $$ \partial_ { \mathbf{W}^ {(l)}} \mathbf{o}= \underbrace{ \partial_ { \mathbf{h}^ {(L-1)}} \mathbf{h}^ {(L)}}_ {\mathbf{M}^ {(L)} \overset{ \mathrm{def}}{ \operatorname*{ = }}} \cdot \ldots \cdot \underbrace{ \partial_ { \mathbf{h}^ {(l)}} \mathbf{h}^ {( l + 1 )}}_ { \mathbf{M}^ {(l + 1)} \overset{ \mathrm{ def }}{ \operatorname * { = }}} \underbrace{ \partial_ { \mathbf{ W }^ {(l)}} \mathbf{h}^ {(l)}}_ { \mathbf{v}^ {(l)} \overset{ \mathrm{ def }}{ \operatorname * {=}}}$$
    - 该梯度是L − l个矩阵与梯度向量$\mathbf{v}^{(l)}$的乘积。因此，我们容易受到数值下溢问题的影响. 当将太多的概率乘在一起时，这些问题经常会出现。在处理概率时，一个常见的技巧是切换到对数空间。不幸的是，上面的问题更为严重：最初，矩阵 M(l) 可能具有各种各样的特征值。他们可能很小，也可能很大；他们的乘积可能非常大，也可能非常小。
    - 不稳定梯度带来的风险不止在于数值表示；也威胁到我们优化算法的稳定性。梯度爆炸（gradient exploding）问题：参数更新过大，破坏了模型的稳定收敛；是梯度消失（gradient vanishing）问题：参数更新过小，在每次更新时几乎不会移动，导致模型无法学习。
### 8.1.1 梯度消失
- sigmoid函数更符合生物神经元的原理，但可能梯度消失：

    


```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from d2l import torch as d2l

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y= torch.sigmoid(x)
y.backward(torch.ones_like(x))

#注意用的是x.grad
d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()] , legend=['sigmoid','gradient'], figsize=(4.5, 2.5))
```

​      ![svg](D:/blog/themes/yilia/source/img/deeplearning/code/pytorch/3_mlp/8_init_files/8_init_2_0.svg)

![svg](img/deeplearning/code/pytorch/3_mlp/8_init_files/8_init_2_0.svg)
    


- 当sigmoid函数输入很大或很小时，他的梯度会消失。此外，当反向传播通过许多层时，除非我们在刚刚好的地方，否则整个乘积的梯度可能会消失。当我们的网络有很多层时，除非我们很小心，否则在某一层可能会切断梯度。因此，更稳定的ReLU系列函数已经成为从业者的默认选择（虽然在神经科学的角度看起来不太合理）。
### 8.1.2 梯度爆炸
- 生成100个高斯随机矩阵，将他们与某个初始矩阵相乘，矩阵乘积会发生梯度爆炸。当这种情况是由于深度网络的初始化所导致时，我们没有机会让梯度下降优化器收敛。


```python
M = torch.normal(0,1, size=(4,4))
print('第一个矩阵\n',M)
for i in range(100):
    M=torch.mm(M, torch.normal(0,1,size=(4,4)))
print('乘以100个矩阵后\n',M)
```

    第一个矩阵
     tensor([[ 0.5260, -0.8163, -0.2948,  1.3394],
            [-0.6026,  0.9170,  0.2718,  1.0166],
            [ 1.3275, -0.1824,  1.1752,  0.2823],
            [ 0.8749,  0.4343,  0.0216,  0.8288]])
    乘以100个矩阵后
     tensor([[ 1.5284e+25,  3.4216e+24, -2.3112e+23, -1.6804e+25],
            [ 5.5275e+24,  1.2374e+24, -8.3584e+22, -6.0772e+24],
            [ 2.6396e+25,  5.9094e+24, -3.9915e+23, -2.9022e+25],
            [ 1.8637e+25,  4.1723e+24, -2.8182e+23, -2.0490e+25]])


### 8.1.3 打破对称性
- 一个隐藏层中的所有隐藏单元的地位是相同的，具有排列对称性。
- 如果将隐藏层的所有参数初始化为相同的常量，那么我们可能永远也无法实现网络的表达能力，隐藏层的行为就好像只有一个单元。虽然小批量随机梯度下降不会打破这种对称性，但暂退法正则化可以。
## 8.2 参数初始化
- 解决或减轻上述问题
### 8.2.1 默认初始化
- 如果我们不指定初始化方法，框架将使用默认的随机初始化方法，对于中等难度的问题，这种方法通常很有效。
### 8.2.2 Xavier初始化
- 假设某层有$n_{in}$个输入：$x_j$，权重为$w_ij$，输出为：$o_i=\sum_{j=1}^{n_{\mathrm{in}}}w_{ij}x_j$
- 假设权重$w_{ij}$服从$ N(0, {\sigma} ^ 2) $，输入$x_j$服从$N(0,\gamma ^2)$。（这并不意味着分布必须是高斯的，只是均值和方差需要存在。）
- 输出$o_i$的均值方差：
    $$\begin{aligned}E[o_{i}] &= \sum_{j=1}^{n_\mathrm{in}}E[w_{ij}x_j]  \\ &= \sum_{j=1}^{n_\mathrm{in}}E[w_{ij}]E[x_j] \\ &= 0.\end{aligned}$$

    $$\begin{aligned}\operatorname{Var}[o_i] &= E[o_i^2]-(E[o_i])^2  \\ &= \sum_{j=1}^{n_{\mathrm{in}}}E[w_{ij}^2x_j^2]-0 \\ &= \sum_{j=1}^{n_{\mathrm{in}}}E[w_{ij}^2]E[x_j^2] \\ &= n_\text{in}\sigma^2\gamma^2.\end{aligned}$$
- 保持方差不变的一种方法是设置$n_{in} \sigma^2=1$。考虑反向传播，使用与前向传播相同的推断，我们可以看到，除非$n_{out} \sigma^2=1$否则梯度的方差可能会增大，我们不可能同时满足这两个条件，因此提出Xavier初始化：
    $$\frac12(n_{\mathrm{in}}+n_{\mathrm{out}})\sigma^2=1 \rightarrow \sigma= \sqrt{\frac{2}{n_{in}+n_{out}}}$$
    - 或者改为从$U\left(-\sqrt{\frac6{n_\mathrm{in}+n_\mathrm{out}}},\sqrt{\frac6{n_\mathrm{in}+n_\mathrm{out}}}\right)$
