---
title: "simple_knn.cu"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.simple-knn.simple_knn.cu"
hexo-path:
---

## 1. 流程

1. 待补充

## 2. 包含的类

| 类名 | 解释 |
|---|---|
| [[3dgs_code_learn/modules/submodules/simple-knn/simple_knn.cu/3dgs_CustomMin\|CustomMin]] | - |
| [[3dgs_code_learn/modules/submodules/simple-knn/simple_knn.cu/3dgs_CustomMax\|CustomMax]] | - |
| [[3dgs_code_learn/modules/submodules/simple-knn/simple_knn.cu/3dgs_MinMax\|MinMax]] | - |

## 3. 包含的函数

| 函数名 | 返回 | 解释 |
|---|---|---|
| [[3dgs_code_learn/functions/submodules/simple-knn/simple_knn.cu/3dgs_prepMorton\|prepMorton]] | __host__ __device__ uint32_t | - |
| [[3dgs_code_learn/functions/submodules/simple-knn/simple_knn.cu/3dgs_coord2Morton\|coord2Morton]] | __host__ __device__ uint32_t | - |
| [[3dgs_code_learn/functions/submodules/simple-knn/simple_knn.cu/3dgs_coord2Morton\|coord2Morton]] | __global__ void | - |
| [[3dgs_code_learn/functions/submodules/simple-knn/simple_knn.cu/3dgs_boxMinMax\|boxMinMax]] | __global__ void | - |
| [[3dgs_code_learn/functions/submodules/simple-knn/simple_knn.cu/3dgs_distBoxPoint\|distBoxPoint]] | __device__ __host__ float | - |
| [[3dgs_code_learn/functions/submodules/simple-knn/simple_knn.cu/3dgs_updateKBest\|updateKBest]] | template<int K> __device__ void | - |
| [[3dgs_code_learn/functions/submodules/simple-knn/simple_knn.cu/3dgs_boxMeanDist\|boxMeanDist]] | __global__ void | - |
| [[3dgs_code_learn/functions/submodules/simple-knn/simple_knn.cu/SimpleKNN/3dgs_knn\|SimpleKNN::knn]] | void | - |
