---
title: "renderCUDA"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.backward.cu.renderCUDA"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| ranges | const uint2* __restrict__ | - |
| point_list | const uint32_t* __restrict__ | - |
| W | int | - |
| H | int | - |
| bg_color | const float* __restrict__ | - |
| points_xy_image | const float2* __restrict__ | - |
| conic_opacity | const float4* __restrict__ | - |
| colors | const float* __restrict__ | - |
| depths | const float* __restrict__ | - |
| final_Ts | const float* __restrict__ | - |
| n_contrib | const uint32_t* __restrict__ | - |
| dL_dpixels | const float* __restrict__ | - |
| dL_invdepths | const float* __restrict__ | - |
| dL_dmean2D | float3* __restrict__ | - |
| dL_dconic2D | float4* __restrict__ | - |
| dL_dopacity | float* __restrict__ | - |
| dL_dcolors | float* __restrict__ | - |
| dL_dinvdepths | float* __restrict__ | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| template <uint32_t C> __global__ void __launch_bounds__(BLOCK_X * BLOCK_Y) | - |

## 3. 操作逻辑

1. 待补充

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_code_learn/files/submodules/diff-gaussian-rasterization/cuda_rasterizer/3dgs_backward.cu\|backward.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
