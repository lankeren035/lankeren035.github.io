---
title: "computeCov2DCUDA"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.backward.cu.computeCov2DCUDA"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| P | int | - |
| means | const float3* | - |
| radii | const int* | - |
| cov3Ds | const float* | - |
| h_x | const float | - |
| h_y | float | - |
| tan_fovx | const float | - |
| tan_fovy | float | - |
| view_matrix | const float* | - |
| opacities | const float* | - |
| dL_dconics | const float* | - |
| dL_dopacity | float* | - |
| dL_dinvdepth | const float* | - |
| dL_dmeans | float3* | - |
| dL_dcov | float* | - |
| antialiasing | bool | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| __global__ void | - |

## 3. 操作逻辑

1. 待补充

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_code_learn/files/submodules/diff-gaussian-rasterization/cuda_rasterizer/3dgs_backward.cu\|backward.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
