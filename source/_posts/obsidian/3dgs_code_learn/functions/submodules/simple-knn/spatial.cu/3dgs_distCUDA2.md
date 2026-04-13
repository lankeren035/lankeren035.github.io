---
title: "distCUDA2"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.simple-knn.spatial.cu.distCUDA2"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| points | const torch::Tensor& | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| torch::Tensor | - |

## 3. 操作逻辑

1. 封装一个c++函数做如下处理：
	1) 获取点云个数：p
	2) 获取点云数据的配置信息：opts
		- 配置信息里面包括  这个张量它在 CPU 还是 GPU， 它的数据类型是什么， 它的内存布局之类，用于后面创建张量，跟输入数据在同一设备上
	 3) 使用opts创建输出张量
	 4) 调用SimpleKNN 命名空间的knn函数，计算每个点与最近邻三个点的平均平方距离：[[source/_posts/obsidian/3dgs_code_learn/functions/submodules/simple-knn/simple_knn.cu/SimpleKNN/3dgs_knn|3dgs_knn]] （这里传入参数时，将点云张量内存变成“连续存储”，因为 PyTorch 的张量有时候内存里不一定是紧挨着排的，而 CUDA/C++ 底层函数通常希望拿到一块规则连续的内存。然后用float指针的形式传入，并用float3确保连续3个float当作一个三维点）：
		   $$
             D_{i}= \frac{d_{i,1}^{2}+d_{i,2}^{2}+d_{i,3}^{2} }{3}
            $$
	5) 返回3knn
## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_spatial.cu|spatial.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
