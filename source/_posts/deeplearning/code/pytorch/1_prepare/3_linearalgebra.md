---
title: 3. 线性代数
date: 2024-2-1 14:00:00
tags: [深度学习,机器学习,pytorch]
categories: [深度学习]
comment: true
toc: true
---
#
<!--more-->
# 3. 线性代数
## 3.1 标量



```python
import torch

x=torch.tensor(3.0)
```

## 3.2 向量


```python
x=torch.arange(4)

#1 取值
print(x[3])

#2 长度
print(len(x))
print(x.shape)
```

    tensor(3)
    4
    torch.Size([4])
    

## 3.3 矩阵



```python
A=torch.arange(20).reshape(5,4) #矩阵用大写字母
print(A)

#1 转置
print(A.T)
```

    tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19]])
    tensor([[ 0,  4,  8, 12, 16],
            [ 1,  5,  9, 13, 17],
            [ 2,  6, 10, 14, 18],
            [ 3,  7, 11, 15, 19]])
    

## 3.4 张量


```python
X= torch.arange(24).reshape(2,3,4)
X
```




    tensor([[[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  9, 10, 11]],
    
            [[12, 13, 14, 15],
             [16, 17, 18, 19],
             [20, 21, 22, 23]]])



## 3.5 张量算法的基本性质


```python
A=torch.arange(20,dtype=torch.float32).reshape(5,4)

#1 复制
B=A.clone()
print(A==B)

#2 Hadamard积⊙ (对应位置相乘)
print(A*B)

#3 张量+标量（所有元素加上标量）
print(A+2)

#4 元素个数
print(A.numel())
```

    tensor([[True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True]])
    tensor([[  0.,   1.,   4.,   9.],
            [ 16.,  25.,  36.,  49.],
            [ 64.,  81., 100., 121.],
            [144., 169., 196., 225.],
            [256., 289., 324., 361.]])
    tensor([[ 2.,  3.,  4.,  5.],
            [ 6.,  7.,  8.,  9.],
            [10., 11., 12., 13.],
            [14., 15., 16., 17.],
            [18., 19., 20., 21.]])
    20
    

## 3.6 降维


```python
X= torch.arange(4,dtype=torch.float32).reshape(2,2)

#1 按轴求和
print(X.sum(axis=0)) #压缩掉第0维

#2 按轴求平均
print(X.mean(axis=0))

#3 非降维求和
sum_X=X.sum(axis=0,keepdims=True)
print(sum_X) #还是二维，只是第0维的长度为1

#4 沿某个轴计算A元素的累积总和
print(A.cumsum(axis=0))
```

    tensor([2., 4.])
    tensor([1., 2.])
    tensor([[2., 4.]])
    tensor([[ 0.,  1.,  2.,  3.],
            [ 4.,  6.,  8., 10.],
            [12., 15., 18., 21.],
            [24., 28., 32., 36.],
            [40., 45., 50., 55.]])
    

## 3.7 点积
- $$<x,y>= x^Ty= \Sigma^{d}_{i=1}x_iy_i$$


```python
x=y=torch.arange(4,dtype=torch.float32)
print(x,y,torch.dot(x,y),sep='\n')
```

    tensor([0., 1., 2., 3.])
    tensor([0., 1., 2., 3.])
    tensor(14.)
    

## 3.8 矩阵-向量积


```python
print(A)
print(x)
torch.mv(A,x) #输出是【5，1】但是1被压缩
```

    tensor([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.],
            [12., 13., 14., 15.],
            [16., 17., 18., 19.]])
    tensor([0., 1., 2., 3.])
    




    tensor([ 14.,  38.,  62.,  86., 110.])



## 3.9 矩阵-矩阵乘法



```python
print(A)
B=torch.ones(4,3)
torch.mm(A,B)
```

    tensor([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.],
            [12., 13., 14., 15.],
            [16., 17., 18., 19.]])
    




    tensor([[ 6.,  6.,  6.],
            [22., 22., 22.],
            [38., 38., 38.],
            [54., 54., 54.],
            [70., 70., 70.]])



## 3.10 范数
- 表示一个向量有多大。这里考虑的大小概念不涉及维度，而是分量的大小。
### 3.10.1 向量范数
- 将向量x映射到标量的函数$f$。
- 向量范数必须满足以下性质：
    - 1）按常数因子α缩放向量的所有元素，其范数也会按相同常数因子的绝对值缩放：
    $$f(\alpha x)=|\alpha|f(x)$$
    - 2）三角不等式：
    $$f(x+y)\leq f(x)+f(y)$$
    - 3）非负性：
    $$f(x)\geq 0$$
    - 4）范数最小为0，当且仅当向量全由0组成：
    $$\forall i,[x]_i=0 \Leftrightarrow f(x)=0 $$

### 3.10.2 L2范数
- 平方和的平方根：
$$||x||=||x||_2 = \sqrt{\sum\limits_{i=1}^{n}x_i^2}$$


```python
u=torch.tensor([3.0,-4.0])
torch.norm(u) #范数
```




    tensor(5.)



### 3.10.3 L1范数
- 绝对值之和
$$||x||_1 = \sum\limits_{i=1}^{n}|x_i|$$
- 与L2范数相比，L1范数受异常值的影响较小


```python
torch.abs(u).sum() #L1范数
```




    tensor(7.)



### 3.10.4 Lp范数
$$||x||_p = (\sum\limits_{i=1}^{n}|x_i|^p)^{\frac{1}{p}}$$

### 3.10.5 Frobenius范数
- 矩阵L2范数
$$||X||_F = \sqrt{\sum\limits_{i=1}^{m}\sum\limits_{j=1}^{n}x_{ij}^2}$$


```python
torch.norm(torch.ones((4,9)))
```




    tensor(6.)



- 目标，或许是深度学习算法最重要的组成部分（除了数据），通常被表达为范数。
## 3.11 练习


```python
a=torch.tensor([[1,2,3],[4,5,6]])
b=a.sum(axis=0)
c=a.sum(axis=1)
print(a.shape,b.shape,c.shape)

print(a+b) #b->(2)->(1,2)->(2,3)
print(a+c) #c->(3)->(1,3)->不匹配
```

    torch.Size([2, 3]) torch.Size([3]) torch.Size([2])
    tensor([[ 6,  9, 12],
            [ 9, 12, 15]])
    


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[37], line 7
          4 print(a.shape,b.shape,c.shape)
          6 print(a+b) #b->(2)->(1,2)->(2,3)
    ----> 7 print(a+c) #c->(3)->(1,3)->不匹配
    

    RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1

