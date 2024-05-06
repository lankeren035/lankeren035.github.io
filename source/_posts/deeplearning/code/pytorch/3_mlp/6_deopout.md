---
title: 3.6 暂退法
date: 2024-2-7 14:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---
#  
<!--more-->
# 6 暂退法
- 模型简单性：
    - 维度小
    - 参数泛化
    - 平滑性
    
- 平滑性：函数不应该对其输入的微小变化敏感。例如，当我们对图像进行分类时，我们预计向像素添加一些随机噪声应该是基本无影响的。具有输入噪声的训练等价于Tikhonov正则化。基于此，提出暂退法：在计算后续层之前向网络的每一层注入噪声。因为当训练一个有多层的深层网络时，注入噪声只会在输入‐输出映射上增强平滑性。从表面上看是在训练过程中丢弃（dropout）一些神经元。

- 需要说明的是，暂退法的原始论文提到了一个关于有性繁殖的类比：神经网络过拟合与每一层都依赖于前一层激活值相关，称这种情况为“共适应性”。作者认为，暂退法会破坏共适应性，就像有性生殖会破坏共适应的基因一样。???  

- 做法：

  <span style="display:block">
  $h ^\prime =\begin{cases}0 &p \\ \frac{h}{1-p} &1-p \\ \end{cases}.$ 

  </span>

- 这样，输出层的计算不能过度依赖于h1, . . . , h5的任何一个元素。通常，我们在测试时不用暂退法。(一些研究人员在测试时使用暂退法，用于估计神经网络预测的“不确定性”：如果通过许多不同的暂退法遮盖后得到的预测结果都是一致的，那么我们可以说网络发挥更稳定。)
 ![svg](D:/blog/themes/yilia/source/img/deeplearning/code/pytorch/3_mlp/6img/1.png)
![](img/deeplearning/code/pytorch/3_mlp/6img/1.png)

## 6.1 代码实现
- 1）从U[0,1]抽样
- 2）保留大于p的点/(1-p)


```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from torch import nn
from d2l import torch as d2l

#1 定义dropout
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float() #大于p的元素->1
    return mask * X / (1.0 - dropout)

#2 测试dropout
X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
```

    tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11., 12., 13., 14., 15.]])
    tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11., 12., 13., 14., 15.]])
    tensor([[ 0.,  2.,  0.,  6.,  0., 10., 12.,  0.],
            [16.,  0.,  0., 22., 24.,  0.,  0.,  0.]])
    tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.]])


### 6.1.1 dropout应用到模型
- 将暂退法应用于每个隐藏层的输出（在激活函数之后），并且可以为每一层分别设置暂退概率：常见的技巧是在靠近输入层的地方设置较低的暂退概率。


```python
#1 参数
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5

#2 模型
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        if self.training: # 只在训练模型时使用dropout
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training: #第二个全连接层后也使用dropout
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out
    
net=Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

#3 训练和测试
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction="none")
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

  ![svg](D:/blog/themes/yilia/source/img/deeplearning/code/pytorch/3_mlp/6_deopout_files/6_deopout_3_0.svg)    
![](img/deeplearning/code/pytorch/3_mlp/6_deopout_files/6_deopout_3_0.svg)    


### 6.1.2 简洁实现



```python
net= nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,256),
    nn.ReLU(), 
    nn.Dropout(dropout1), #dropout
    nn.Linear(256,256),
    nn.ReLU(),
    nn.Dropout(dropout2), #dropout
    nn.Linear(256,10)
                   )
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

# 训练和测试
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

  ![svg](D:/blog/themes/yilia/source/img/deeplearning/code/pytorch/3_mlp/6_deopout_files/6_deopout_5_0.svg)
    
![](img/deeplearning/code/pytorch/3_mlp/6_deopout_files/6_deopout_5_0.svg)
    

