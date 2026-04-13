---
title: "updateKBest"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.simple-knn.simple_knn.cu.updateKBest"
hexo-path:
---
- 输入： \[5,10,20] , 7  -> \[5,7,20] , 10 -> \[5,7,10], 20
## 1. 输入

| 参数名   | 类型            | 解释            |
| ----- | ------------- | ------------- |
| ref   | const float3& | 当前点           |
| point | const float3& | 候选点           |
| knn   | float*        | 当前保存的K个最近邻的距离 |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| template<int K> __device__ void | - |

## 3. 操作逻辑

1.  输入： \[5,10,20] , 7  -> \[5,7,20] , 10 -> \[5,7,10], 20

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_simple_knn.cu\|simple_knn.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
