---
title: 1.5 自动微分
date: 2024-2-1 14:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---
#  
<!--more-->
# 5. 自动微分
- 深度学习框架通过自动计算导数，即自动微分（automatic differentiation）来加快求导。根据设计好的模型，系统会构建一个计算图（computational graph），来跟踪计算是哪些数据通过哪些操作组合起来产生输出。自动微分使系统能够随后反向传播梯度。这里，反向传播（backpropagate）意味着跟踪整个计算图，填充关于每个参数的偏导数。
## 5.1 
- 对函数$y=2\mathbf{x}^{\top}\mathbf{x}$关于列向量$\mathbf{x}$求导


```python
import torch
#1 创建变量x
x=torch.arange(4.0)
y=2*torch.dot(x,x)
print(x)

#2 存储梯度 （一个标量函数关于向量x的梯度是向量，并且与x具有相同的形状。）
x.requires_grad_(True) #等价于x=torch.arange(4.0,requires_grad=True)
print(x)

#3 计算y
y=2*torch.dot(x,x) #点积
print(y)

#4 反向传播计算梯度
print(x.grad) #默认为None
y.backward() #反向传播
print(x.grad) #梯度为4x

#5 阻止跟踪
y=x.sum() #y=6
y.backward()
print(x.grad) #x的梯度会累加
x.grad.zero_() #清除梯度后就不会累加了
y.backward()
print(x.grad)
```

    tensor([0., 1., 2., 3.])
    tensor([0., 1., 2., 3.], requires_grad=True)
    tensor(28., grad_fn=<MulBackward0>)
    None
    tensor([ 0.,  4.,  8., 12.])
    tensor([ 1.,  5.,  9., 13.])
    tensor([1., 1., 1., 1.])
    

## 5.2 非标量变量的反向传播



```python
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y=x*x #这里y是向量
y.sum().backward() #这里y.sum()是标量,所以可以调用backward，相当于y.backward(torch.ones(len(x)))
print(x.grad)

```

    tensor([0., 2., 4., 6.])
    

## 5.3 分离计算
- 将某些计算移动到记录的计算图之外。（作为常数处理）
    - 假设$y=f(x), z=g(x,y)$。我们想计算z关于x的梯度，但由于某种原因，希望将y视为一个常数，并且只考虑到x在y被计算后发挥的作用。


```python
x.grad.zero_()
y=x*x
u=y.detach() #分u相当于一个常数，不需要求梯度
z=u*x
z.sum().backward()
print(x.grad) #u

x.grad.zero_()
z=y*x
z.sum().backward()
print(x.grad) #3x^2
```

    tensor([0., 1., 4., 9.])
    tensor([ 0.,  3., 12., 27.])
    

## 5.4 python控制流的梯度计算


```python
def f(a):
    b=a*2
    while b.norm()<1000: #经过while后b=ka
        b=b*2
    if b.sum()>0:
        c=b #c=kb=ka
    else:
        c=100*b #c=100kb=100ka
    return c #ka
a=torch.randn(size=(),requires_grad=True)
D=f(a) #
D.backward()
a.grad==D/a #梯度就是k=D/a
```




    tensor(True)


