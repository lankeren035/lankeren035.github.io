---
title: "distBoxPoint"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.simple-knn.simple_knn.cu.distBoxPoint"
hexo-path:
---
- 计算一个点距离一个盒子的距离，如果这个点在盒子内，距离=0， 如果点在盒子外但是x范围在盒子内，则x轴距离=0.
## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| box | const MinMax& | - |
| p | const float3& | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| __device__ __host__ float | - |

## 3. 操作逻辑

1. 计算一个点距离一个盒子的距离，如果这个点在盒子内，距离=0， 如果点在盒子外但是x范围在盒子内，则x轴距离=0.

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_simple_knn.cu\|simple_knn.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
