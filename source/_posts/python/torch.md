---
title: python中torch的使用
date: 2024-5-9 11:00:00
tags: [python,pytorch]
categories: [python]
comment: true
toc: true
---
#

<!--more-->
# python中torch的使用

- 初始化：
```python
import torch.nn as nn

def xavier_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])

net.apply(xavier_init_weights)
```

- 生成矩阵：
```python

torch.eye(10).unsqueeze(0).unsqueeze(0).repeat(2, 2, 1, 1)
```