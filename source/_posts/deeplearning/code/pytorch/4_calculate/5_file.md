---
title: 4.5 读写文件
date: 2024-4-20 16:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---
#  
<!--more-->
# 5 读写文件
## 5.1 加载和保存张量



```python
import torch
from torch import nn
from torch.nn import functional as F

# 保存一个张量
x = torch.arange(4)
torch.save(x,'x-file') # 保存，二进制文件
x2 = torch.load('x-file') #加载
print(x2)

# 保存一个张量列表
y = torch.zeros(4)
torch.save([x,y], 'x-files')
x2, y2 = torch.load('x-files')
print(x2,y2)

# 保存字符串-张量字典
mydict = {'x':x, 'y':y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)
```

    tensor([0, 1, 2, 3])
    tensor([0, 1, 2, 3]) tensor([0., 0., 0., 0.])
    {'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}
    

## 5.2 加载和保存模型参数


```python
class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))
net = MLP()
X = torch.randn(size=(2,20))
Y = net(X)

#保存模型
torch.save(net.state_dict(), 'mlp.params')

#加载模型
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())

#验证加载后的模型是否与初始模型相等
Y_clone = clone(X)
print(Y_clone == Y)
```

    MLP(
      (hidden): Linear(in_features=20, out_features=256, bias=True)
      (output): Linear(in_features=256, out_features=10, bias=True)
    )
    tensor([[True, True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True, True]])
    
