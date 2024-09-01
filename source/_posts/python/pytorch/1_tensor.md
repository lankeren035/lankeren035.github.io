---
title: 1- 张量
date: 2024-9-1 10:00:00
tags: [pytorch,python]
categories: [pytorch]
comment: true
toc: true

---

#

<!--more-->

# 1- 张量

## 1.1 创建张量

### 1）根据列表

```python
import torch

data = [[1,2],[3,4]]
x_data = torch.tensor(data)

print(type(x_data))
```

### 2）根据numpy数组

```python
import torch
import numpy as np

data = np.array([[1,2],[3,4]])
x_data = torch.from_numpy(data)

print(type(x_data))
```

### 3）根据Tensor

```python
import torch
data = [[1,2],[3,4]]
x_data = torch.tensor(data)

x_ones = torch.ones_like(x_data)
x_zeros = torch.zeros_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float) #rand默认生成浮点数

print(f"张量1： {x_ones}")
print(f"张量0： {x_zeros}")
print(f"张量随机： {x_rand}")
```

### 4）根据形状

```python
shape=(2,3)

x_ones = torch.ones(shape)
x_zeros = torch.zeros(shape)
x_rand = torch.rand(shape) #rand默认生成浮点数

print(f"张量1： {x_ones}")
print(f"张量0： {x_zeros}")
print(f"张量随机： {x_rand}")
```

## 1.2 张量的属性

| 属性   | 解释         |
| ------ | ------------ |
| shape  | 形状         |
| dtype  | 数据类型     |
| device | 在哪个设备上 |

## 1.3 张量操作

### 1.3.1 张量判断

| 操作                | 解释         |
| ------------------- | ------------ |
| is_tensor()         | 是否为tensor |
| is_complex()        | 元素为复数   |
| is_floating_point() | 元素为浮点数 |
| numel()             | 元素个数     |

### 1.3.2 创建操作

| 操作                                                         | 解释                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <a href=#tensor>tensor()</a>                                 | 根据内容创建张量                                             |
| <a href=#zeros>zeros()</a><br>ones()<br>eye()<br>empty()<br>full(形状, 数值) | 全0矩阵<br>全1矩阵<br>单位矩阵<br>未初始化的张量<br>单元素填充 |
| <a href=#zeroslike>zeros_like()</a><br>ones_like()           | 全0矩阵<br>全1矩阵                                           |
| <a href=#arange>arange()</a>                                 | 范围：[   )                                                  |
| <a href=#range>range()</a>                                   | 范围：[   ]，steps指定步长                                   |
| <a href=#linspace>linspace()</a><br>logspace                 | 范围：[   ]，steps指定个数<br>确定底数，生成一系列指数       |

### 1.3.



> - <a id=tensor></a> torch.tensor(*data*, ***, *dtype=None*, *device=None*, *requires_grad=False*, *pin_memory=False*) → Tensor

- 

> - <a id=zeros></a> torch.zeros(**size*, ***, *out=None*, *dtype=None*, *layout=torch.strided*, *device=None*, *requires_grad=False*) → Tensor

```python
import torch

x = torch.zeros(2,3)

print(x)
```

> - <a id=zeroslike></a> torch.zeros_like(*input*, ***, *dtype=None*, *layout=None*, *device=None*, *requires_grad=False*, *memory_format=torch.preserve_format*) → Tensor

```python
import torch

x = torch.ones(2,3)
y = torch.zeros_like(x)

print(y)
```

> - <a id=arange></a> torch.arange(*start=0*, *end*, *step=1*, ***, *out=None*, *dtype=None*, *layout=torch.strided*, *device=None*, *requires_grad=False*) → Tensor

```python
import torch

x = torch.arange(5)

print(x)
```

> - <a id=range></a> torch.range(*start=0*, *end*, *step=1*, ***, *out=None*, *dtype=None*, *layout=torch.strided*, *device=None*, *requires_grad=False*) → Tensor

```python
import torch

x = torch.range(start=0, end=5)

print(x)
```

>-  <a id=linspace></a> torch.linspace(*start*, *end*, *steps*, ***, *out=None*, *dtype=None*, *layout=torch.strided*, *device=None*, *requires_grad=False*) → Tensor

```python
import torch

x = torch.linspace(start=0, end=5,steps=3)

print(x)
```

> -  torch.logspace(*start*, *end*, *steps*, *base=10.0*, ***, *out=None*, *dtype=None*, *layout=torch.strided*, *device=None*, *requires_grad=False*) → Tensor
>

```python
import torch

x = torch.logspace(start=-10, end=10, steps=5)# 底数默认10

print(x)
```

