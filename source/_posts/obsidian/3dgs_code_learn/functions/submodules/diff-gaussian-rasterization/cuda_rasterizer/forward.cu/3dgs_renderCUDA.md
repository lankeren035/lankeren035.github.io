---
title: "renderCUDA"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.forward.cu.renderCUDA"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| ranges | const uint2* __restrict__ | - |
| point_list | const uint32_t* __restrict__ | - |
| W | int | - |
| H | int | - |
| points_xy_image | const float2* __restrict__ | - |
| features | const float* __restrict__ | - |
| conic_opacity | const float4* __restrict__ | - |
| final_T | float* __restrict__ | - |
| n_contrib | uint32_t* __restrict__ | - |
| bg_color | const float* __restrict__ | - |
| out_color | float* __restrict__ | - |
| depths | const float* __restrict__ | - |
| invdepth | float* __restrict__ | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| template <uint32_t CHANNELS> __global__ void __launch_bounds__(BLOCK_X * BLOCK_Y) | - |

## 3. 操作逻辑

1. 待补充

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_forward.cu\|forward.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
