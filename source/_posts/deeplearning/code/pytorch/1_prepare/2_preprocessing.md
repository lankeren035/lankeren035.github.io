---
title: 2. pytorch数据预处理
date: 2023-11-27 14:00:00
tags: [深度学习,机器学习,pytorch,数据预处理]
categories: [深度学习]
comment: false
toc: true
---
# 
<!--more-->

# 2. pytorch数据预处理
- 主要通过pandas预处理

## 2.1 读取数据集


```python
# 1. 先自己准备一个数据集
import os
os.makedirs(os.path.join('..','data'),exist_ok=True)
data_file=os.path.join('..','data','house_tiny.csv')
with open(data_file,'w') as f:
    f.write('NumRooms, Alley, Price\n') #列名
    f.write('NA,Pave, 127500\n') #每行表示一个数据样本
    f.write('2,NA, 106000\n')
    f.write('4,NA, 178100\n')
    f.write('NA,NA, 14000\n')

# 2. 读取数据集
import pandas as pd
data=pd.read_csv(data_file)
print(data)
```

       NumRooms  Alley   Price
    0       NaN   Pave  127500
    1       2.0    NaN  106000
    2       4.0    NaN  178100
    3       NaN    NaN   14000
    

## 2.2 缺失值处理
- 删除
- 插值

### 2.2.1 数值类型


```python
# 1. 位置索引iloc将data分为输入与输出
inputs=data.iloc[:,0:2] #前两列作为输入
outputs=data.iloc[:,2] #第三列作为输出

# 2. 用每一列的均值替换空值
inputs=inputs.fillna(inputs.mean())
print(inputs)
```

       NumRooms  Alley
    0       3.0   Pave
    1       2.0    NaN
    2       4.0    NaN
    3       3.0    NaN
    

### 2.2.2 类别类型或离散值
- 将NaN视为一个类别，根据这一列类别的个数分出n列，每一列代表一个类别，如果该行的值为该列的类别，则为1，否则为0


```python
inputs=pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

       NumRooms   Alley_Pave   Alley_nan
    0       3.0            1           0
    1       2.0            0           1
    2       4.0            0           1
    3       3.0            0           1
    

## 2.3 转换为张量
- torch.tensor()


```python
import torch
x=torch.tensor(inputs.to_numpy(dtype=float))
y=torch.tensor(outputs.to_numpy(dtype=float))
x,y
```




    (tensor([[3., 1., 0.],
             [2., 0., 1.],
             [4., 0., 1.],
             [3., 0., 1.]], dtype=torch.float64),
     tensor([127500., 106000., 178100.,  14000.], dtype=torch.float64))



## 练习
- 1. 创建一个更多行和列的数据集



```python
import os
os.makedirs(os.path.join('..','data'),exist_ok=True)
data_file=os.path.join('..','data','house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms, Alley, Price, test\n') # 列名
    f.write('NA, Pave, 127500, 1\n') # 每行表示一个数据样本
    f.write('2, NA, 106000, 0\n')
    f.write('4, NA, 178100, 0\n')
    f.write('NA, NA, 140000, 1\n')
    f.write('2, Pave, 127500, 0\n')

#读取数据集
import pandas as pd
print(pd.read_csv(data_file))
```

       NumRooms  Alley   Price   test
    0       NaN   Pave  127500      1
    1       2.0     NA  106000      0
    2       4.0     NA  178100      0
    3       NaN     NA  140000      1
    4       2.0   Pave  127500      0
    

- 2. 删除缺失值最多的列


```python
#删除缺失值最多的列
import pandas as pd
data=pd.read_csv(data_file)

#计算每一列的缺失值个数
missing=data.isnull().sum() #按列求和
print(missing)
column=missing.idxmax() #返回缺失值最多的列名
print(column)
data=data.drop(columns=[column])
print(data)

#保存处理后的数据集
data.to_csv(data_file,index=False) #index=False表示不保存行索引
```

    NumRooms    2
     Alley      0
     Price      0
     test       0
    dtype: int64
    NumRooms
       Alley   Price   test
    0   Pave  127500      1
    1     NA  106000      0
    2     NA  178100      0
    3     NA  140000      1
    4   Pave  127500      0
    

- 3. 将处理后的数据集转换为张量


```python
#将数据集转换为张量格式
import torch
import os
data_file=os.path.join('..','data','house_tiny.csv')
data=pd.read_csv(data_file)
#输出data的shape
print(data,data.shape,sep='\n')

#将字符串类型进行one-hot编码
data=pd.get_dummies(data,dummy_na=True) #dummy_na=True表示将缺失值也当作合法的特征值并为其创建指示特征
print(data)
#将dataframe格式转换为张量格式
data=torch.tensor(data.to_numpy(dtype=float))
print(data)
```

       Alley   Price   test
    0   Pave  127500      1
    1     NA  106000      0
    2     NA  178100      0
    3     NA  140000      1
    4   Pave  127500      0
    (5, 3)
        Price   test   Alley_ NA   Alley_ Pave   Alley_nan
    0  127500      1           0             1           0
    1  106000      0           1             0           0
    2  178100      0           1             0           0
    3  140000      1           1             0           0
    4  127500      0           0             1           0
    tensor([[1.2750e+05, 1.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00],
            [1.0600e+05, 0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00],
            [1.7810e+05, 0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00],
            [1.4000e+05, 1.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00],
            [1.2750e+05, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00]],
           dtype=torch.float64)
    
