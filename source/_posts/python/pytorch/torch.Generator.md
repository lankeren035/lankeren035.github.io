---
title: torch.Generator
date: 2025-1-17 10:00:00
tags: [pytorch,python]
categories: [pytorch]
comment: true
toc: true


---

#

<!--more-->



# 0. 使用默认的随机数生成器

```python
import torch

# 设置默认的随机数种子
torch.manual_seed(0)

# 查看默认的随机数种子
torch.initial_seed()
```



# 1. 指定随机数生成器

- 指定默认随机数生成器（类似上一节）

  ```python
  # 获取默认的 torch.Generator 实例
  g_1 = torch.default_generator
  
  # 查看指定随机数生成器的种子(结果也为 0)
  g_1.initial_seed()
  ```

- 手动创建生成器

  ```python
  # 1. 使用默认随机数生成器
  torch.manual_seed(1)
  
  # 结果 tensor([0, 4, 2, 3, 1])
  torch.randperm(5)
  
  
  # 2. 手动创建随机数生成器
  g = torch.Generator()
  g.manual_seed(1)
  
  # 结果也为 tensor([0, 4, 2, 3, 1])
  torch.randperm(5, generator=g)
  
  ```

  

# 2. 在GPU上使用随机数生成器

- 给当前 GPU 设备的默认随机数生成器设置种子

  ```python
  torch.cuda.manual_seed(0)
  torch.cuda.initial_seed()
  ```

- 获取 GPU 设备的默认随机数生成器 

  ```python
  torch.cuda.default_generators #因为一台电脑可以有多个 GPU 设备, 所以返回了 torch.Generator 元组
  ```

- 实例化一个 GPU 类型的随机数生成器

  ```python
  import torch
  
  g = torch.Generator(device='cuda:1')
  g.manual_seed(1)
  
  t = torch.randperm(5, device='cuda:0', generator=g)
  print(t)
  ```

  