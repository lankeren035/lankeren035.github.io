---
title: "computeCov2D"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.forward.cu.computeCov2D"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| mean | const float3& | - |
| focal_x | float | - |
| focal_y | float | - |
| tan_fovx | float | - |
| tan_fovy | float | - |
| cov3D | const float* | - |
| viewmatrix | const float* | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| __device__ float3 | - |

## 3. 操作逻辑

1. 待补充

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_code_learn/files/submodules/diff-gaussian-rasterization/cuda_rasterizer/3dgs_forward.cu\|forward.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
