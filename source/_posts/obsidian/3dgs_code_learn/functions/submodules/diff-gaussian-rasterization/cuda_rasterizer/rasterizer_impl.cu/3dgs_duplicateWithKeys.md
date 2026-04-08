---
title: "duplicateWithKeys"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.rasterizer_impl.cu.duplicateWithKeys"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| P | int | - |
| points_xy | const float2* | - |
| depths | const float* | - |
| offsets | const uint32_t* | - |
| gaussian_keys_unsorted | uint64_t* | - |
| gaussian_values_unsorted | uint32_t* | - |
| radii | int* | - |
| grid | dim3 | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| __global__ void | - |

## 3. 操作逻辑

1. 待补充

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_code_learn/files/submodules/diff-gaussian-rasterization/cuda_rasterizer/3dgs_rasterizer_impl.cu\|rasterizer_impl.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
