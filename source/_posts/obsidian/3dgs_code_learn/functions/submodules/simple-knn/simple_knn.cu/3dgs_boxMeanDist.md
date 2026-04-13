---
title: "boxMeanDist"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.simple-knn.simple_knn.cu.boxMeanDist"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| P | uint32_t | - |
| points | float3* | - |
| indices | uint32_t* | - |
| boxes | MinMax* | - |
| dists | float* | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| __global__ void | - |

## 3. 操作逻辑

1. 前面已经做过morton排序了，对每个点，取它左右各三个邻居进行比较，得到一个粗糙的3knn（里面记录最近的三个点的距离） [[3dgs_updateKBest]]
2. 将该3knn\[2]作为判据，如果某个盒子与当前点[[3dgs_distBoxPoint|距离]]大于这个值，则跳过该盒子，否则遍历该盒子的点，更新3knn[[3dgs_updateKBest]]
3. 返回：$D_{i}= \frac{d_{i,1}^{2}+d_{i,2}^{2}+d_{i,3}^{2} }{3}$

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_simple_knn.cu\|simple_knn.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
