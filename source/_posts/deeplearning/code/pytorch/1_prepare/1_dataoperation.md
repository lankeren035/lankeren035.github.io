---
title: 1.1 pytorch数据操作
date: 2023-11-11 14:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: false
toc: true
---
#  
<!--more-->


# 1. pytorch数据操作

- pytorch与tensorflow中的Tensor类似于numpy的ndarray
- pytorch中的Tensor可以在GPU上运行
- pytorch中的Tensor可以用于自动求导



## 1.1 张量

- 创建张量


```python
# 1. 导入torch，不是pytorch
import torch

# 2. 创建范围张量
x=torch.arange(12) # 0-11,默认为int64，可以指定dtype，默认存储在CPU上
y=torch.arange(12,dtype=torch.float32) # 指定dtype

# 3. 创建全0张量
zeros=torch.zeros(2,3,4) # 2*3*4的全0张量

# 4. 创建全1张量
ones=torch.ones(2,3,4) # 2*3*4的全1张量

# 5. 创建采样张量
sample=torch.randn(3,4) #从标准高斯分布中采样

# 6. 列表张量
lists=torch.tensor([[1,2,3],[4,5,6]])


print(x)
print(y)
print(zeros)
print(ones)
print(sample)
print(lists)

```

    tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
    tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.])
    tensor([[[0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.]],
    
            [[0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.]]])
    tensor([[[1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.]],
    
            [[1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.]]])
    tensor([[-0.0847, -0.2406,  0.1735, -1.5543],
            [-0.2820, -0.6689,  0.0565,  0.4746],
            [ 0.9841, -1.6116, -0.1587, -1.5121]])
    tensor([[1, 2, 3],
            [4, 5, 6]])
    

- 张量的属性


```python
# 3. 查看张量的形状
print('x的规模：',x.shape)

# 4. 查看张量的元素总数
print('x中元素个数：',x.numel())

# 5. 改变张量的形状
x=x.reshape(3,4)
print('改变x的形状：',x)
print('自动计算的形状：',x.reshape(-1,6)) # -1表示自动计算这个维度
```

    x的规模： torch.Size([12])
    x中元素个数： 12
    改变x的形状： tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]])
    自动计算的形状： tensor([[ 0,  1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10, 11]])
    

## 1.2 运算符


```python
x=torch.arange(4)

# 1. 按元素运算
print('按元素加法：',x+x)
print('按元素减法：',x-x)
print('按元素乘法：',x*x)
print('按元素除法：',x/x)
print('按元素幂运算：',x**x)
print('按元素开方：',x.sqrt())
print('求幂运算：',torch.exp(x))

# 2. 矩阵运算

# 3. 合并张量(记得加括号)
x=torch.zeros(12).reshape(3,4)
y=torch.ones(12).reshape(3,4)
print('沿行合并：',torch.cat((x,y),dim=0)) # 沿行合并
print('沿列合并：',torch.cat((x,y),dim=1)) # 沿列合并

# 4. 逻辑运算
print(x==y)

# 5. 求和
print('求和：',y.sum())
```

    按元素加法： tensor([0, 2, 4, 6])
    按元素减法： tensor([0, 0, 0, 0])
    按元素乘法： tensor([0, 1, 4, 9])
    按元素除法： tensor([nan, 1., 1., 1.])
    按元素幂运算： tensor([ 1,  1,  4, 27])
    按元素开方： tensor([0.0000, 1.0000, 1.4142, 1.7321])
    求幂运算： tensor([ 1.0000,  2.7183,  7.3891, 20.0855])
    沿行合并： tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]])
    沿列合并： tensor([[0., 0., 0., 0., 1., 1., 1., 1.],
            [0., 0., 0., 0., 1., 1., 1., 1.],
            [0., 0., 0., 0., 1., 1., 1., 1.]])
    tensor([[False, False, False, False],
            [False, False, False, False],
            [False, False, False, False]])
    求和： tensor(12.)
    

## 1.3 广播机制
- 对两个形状不同的张量按元素运算时，可能会触发广播机制：先适当复制元素使这两个张量形状相同后再按元素运算

1. 如果两个张量的维度数不同，可以在较小的张量的形状前面补1，直到两者的维度数相同。
2. 如果两个张量在某个维度上的大小不同，但其中一个张量在该维度上的大小为1，那么可以通过在该维度上重复扩展该张量，使得两个张量在该维度上的大小相同。
3. 如果两个张量在某个维度上的大小都不为1，且大小不同，那么会发生形状不匹配，导致无法进行广播

```python
a=torch.arange(3).reshape(3,1)
b=torch.arange(2).reshape(1,2)
print('a:',a)
print('b:',b)
print('a+b:',a+b) #通常沿着长度为1的维度进行广播
```

    a: tensor([[0],
            [1],
            [2]])
    b: tensor([[0, 1]])
    a+b: tensor([[0, 1],
            [1, 2],
            [2, 3]])
    

## 1.4 索引和切片


```python
print('x:',x)
print('x[-1]:',x[-1])
print('x[1:3]:',x[1:3]) #选择第2和第3个元素
x[1,2]=9 # 修改元素
print('x[1,2]=9后 x:',x)
x[0:2,:]=12 # 修改一行
print('x[0:2,:]=12后 x:',x)
```

    x: tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]])
    x[-1]: tensor([0., 0., 0., 0.])
    x[1:3]: tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.]])
    x[1,2]=9后 x: tensor([[0., 0., 0., 0.],
            [0., 0., 9., 0.],
            [0., 0., 0., 0.]])
    x[0:2,:]=12后 x: tensor([[12., 12., 12., 12.],
            [12., 12., 12., 12.],
            [ 0.,  0.,  0.,  0.]])
    

## 1.5 节省内存


```python
z=torch.zeros(1)
before=id(z)
z=z+x #对于这种类似列表的数据结构，加号会导致新的内存分配
print('z:',z)
print('id(z)==before:',id(z)==before) # False，说明z指向了新的地址

z=torch.zeros(3,4) #使用如下操作可以避免新的内存分配，但是无法广播，所以这里要求x和z的形状一致
before=id(z)
z+=x # +=不会导致新的内存分配
z[:]=x+z # 也不会导致新的内存分配
print('id(z)==before:',id(z)==before) # True
```

    z: tensor([[12., 12., 12., 12.],
            [12., 12., 12., 12.],
            [ 0.,  0.,  0.,  0.]])
    id(z)==before: False
    id(z)==before: True
    

## 1.6 转换为其他python对象
- tensor与numpy数组


```python
A=x.numpy() # 将张量转换为numpy数组
B=torch.tensor(A) # 将numpy数组转换为张量
print(A,B,sep='\n')

A[1,1]=3
print(A,B,x,sep='\n') #A与X共享内存
```

    [[12. 12. 12. 12.]
     [12.  3. 12. 12.]
     [ 0.  0.  0.  0.]]
    tensor([[12., 12., 12., 12.],
            [12.,  3., 12., 12.],
            [ 0.,  0.,  0.,  0.]])
    [[12. 12. 12. 12.]
     [12.  3. 12. 12.]
     [ 0.  0.  0.  0.]]
    tensor([[12., 12., 12., 12.],
            [12.,  3., 12., 12.],
            [ 0.,  0.,  0.,  0.]])
    tensor([[12., 12., 12., 12.],
            [12.,  3., 12., 12.],
            [ 0.,  0.,  0.,  0.]])
    

- 张量转为python标量


```python
a=torch.tensor([1.3])
print(a)

#由于pytorch中小数用32位，python中用64位；尝试输出.20f就能发现精确度问题
#由于舍入（四舍五入到最接近的偶数），1.3在pytorch中表示是1.299多，而在python中是1.300多
print(a.item()) #使用item函数
print(float(a)) #python内置函数
```

    tensor([1.3000])
    1.2999999523162842
    1.2999999523162842
    
