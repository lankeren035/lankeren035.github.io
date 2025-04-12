---
title: 'pytorch中register_buffer函数'
date: 2025-3-1 10:28:00
tags: [深度学习,代码]
categories: [pytorch,函数]
comment: false
toc: true

---

#
<!--more-->

```python
import torch.nn as nn
class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale
```

-  `register_buffer`的作用是将  一个tensor变量注册到模型的 buffers() 属性中， 该变量 不会有梯度传播给它，但是能被模型的state_dict记录下来。可以理解为模型的**常数**。 

-  既然register_buffer的对象是模型中的常数，那为什么不直接使用下面的方法一，还不更直接吗 ？

  ```python
  class net(nn.Module):
      def __init__(self,x=None):
          super(net,self).__init__()
  		self.a=torch.ones(2,3)#方法一
  		self.register_buffer("a",torch.ones(2,3))#方法二
  
  ```

  

-  我们可能会遇到这样的场景：那个常数不是这么简单的常数，而是外部传入的。 

  ```python
  class net(nn.Module):
      def __init__(self,x=None):
          super(net,self).__init__()
  		self.a=x#方法一
  		self.register_buffer("a",x)#方法二
  
  x=**
  x=***
  x=**
  #第一次运行的时候，你经过千辛万苦得到了模型中的常数x。
  model=net(x)
  #训练模型
  #保存模型。
  #完毕
  
  ```

  ```python
  #如果是方法一，你又要运行一遍获得x的过程。
  x=**
  x=***
  x=**
  model=net(x)
  #载入模型model.load
  #使用模型
  
  ```

  ```python
  #如果是方法二，不需要获得x，因为register_buffer会将常数x保存在state_dict中，载入就行了。
  model=net(x)
  #载入模型model.load
  #使用模型
  
  ```

  

| 特性             | 说明                                              |
| ---------------- | ------------------------------------------------- |
| **非可训练参数** | 不会被梯度更新，但参与前向传播                    |
| **设备感知**     | 自动随模型切换设备（CPU/GPU）                     |
| **状态持久化**   | 会被保存到`state_dict`中，随模型参数一起保存/加载 |
| **广播优化**     | 通过预设维度对齐，实现张量自动广播                |