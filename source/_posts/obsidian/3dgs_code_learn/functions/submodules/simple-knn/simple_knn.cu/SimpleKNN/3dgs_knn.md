---
title: "knn"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.simple-knn.simple_knn.cu.SimpleKNN.knn"
hexo-path:
---

## 1. 输入

| 参数名       | 类型      | 解释      |
| --------- | ------- | ------- |
| P         | int     | 点云中点的个数 |
| points    | float3* | 点云张量    |
| meanDists | float*  | 输出张量    |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| void | - |

## 3. 操作逻辑

1. 求整个点云的全局最大/最小坐标（相当于整个场景的边界角）[[3dgs_CustomMin]][[3dgs_CustomMax]]
2. 将每个点进行Morton编码[[3dgs_coord2Morton]]
3. 将点按Morton编码排序
4. 每1024个点分到一个box，并给每个box算包围盒（最大最小坐标）[[3dgs_boxMinMax]]
5. 对每个点找3个最近邻，输出：$D_{i}= \frac{d_{i,1}^{2}+d_{i,2}^{2}+d_{i,3}^{2} }{3}$ [[3dgs_boxMeanDist]]

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_simple_knn.cu\|simple_knn.cu]] |
| 所属类 | - |
| 命名空间 | SimpleKNN |
| 类型 | namespace_scoped |
