---
title: 10.1 优化和深度学习
date: 2024-7-16 12:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---
#  
<!--more-->
# 1 优化和深度学习

- 在优化中，损失函数通常被称为优化问题的目标函数。

- 本质上，优化和深度学习的目标是根本不同的。

    - 优化主要关注最小化目标，优化的目标是减少训练误差。
    - 深度学习关注在给定有限数据量的情况下寻找合适的模型。深度学习的目标是减少泛化误差，因此除了使用优化算法来减少误差，我们还需要注意过拟合。


```python
%matplotlib inline
import numpy as np
import torch
from mpl_toolkits import mplot3d
from d2l import torch as d2l
```

- 经验风险是训练数据集的平均损失
- 风险是整个数据群的预期损失


```python
# 风险函数
def f(x):
    return x* torch.cos(np.pi * x)

# 经验风险函数
def g(x): # 假设我们只有有限的寻来你数据。因此g不如f平滑
    return f(x) + 0.2 * torch.cos(5 * np.pi * x)

def annotate(text, xy, xytext): #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext, arrowprops=dict(arrowstyle='->'))

x = torch.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))

```


    
![svg](img/deeplearning/code/pytorch/10_optimize/1_optimize_files/1_optimize_3_0.svg)
    

## 1.2 深度学习中的优化挑战

### 1.2.1 局部最小值


```python
x = torch.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))

```


    
![svg](img/deeplearning/code/pytorch/10_optimize/1_optimize_files/1_optimize_5_0.svg)
    


- 只有一定程度的噪声可能会使参数跳出局部最小值。事实上，这是小批量随机梯度下降的有利特性之一。在这种情况下，小批量上梯度的自然变化能够将参数从局部极小值中跳出。

### 1.2.2 鞍点

- 除了局部最小值之外，鞍点是梯度消失的另一个原因。


```python
x = torch.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))

```


    
![svg](img/deeplearning/code/pytorch/10_optimize/1_optimize_files/1_optimize_7_0.svg)
    


- 较高维度的鞍点甚至更加隐蔽。


```python
x, y = torch.meshgrid(
torch.linspace(-1.0, 1.0, 101), torch.linspace(-1.0, 1.0, 101))
z = x**2 - y**2
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');

```

    c:\Users\18048\.conda\envs\d2l\lib\site-packages\torch\functional.py:478: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\native\TensorShape.cpp:2895.)
      return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
    


    
![svg](img/deeplearning/code/pytorch/10_optimize/1_optimize_files/1_optimize_9_1.svg)
    


### 1.2.3 梯度消失

- 可能遇到的最隐蔽问题是梯度消失。

- 例如，假设我们想最小化函数f(x) = tanh(x)，然后我们恰好从x = 4开始。f的梯度接近零。因此，在我们取得进展之前，优化将会停滞很长一段时间。


```python
x = torch.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [torch.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))

```


    
![svg](img/deeplearning/code/pytorch/10_optimize/1_optimize_files/1_optimize_11_0.svg)
    

