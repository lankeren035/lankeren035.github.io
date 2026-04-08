---
title: "render"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.backward.h.BACKWARD.render"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| grid | const dim3 | - |
| block | dim3 | - |
| ranges | const uint2* | - |
| point_list | const uint32_t* | - |
| W | int | - |
| H | int | - |
| bg_color | const float* | - |
| means2D | const float2* | - |
| conic_opacity | const float4* | - |
| colors | const float* | - |
| depths | const float* | - |
| final_Ts | const float* | - |
| n_contrib | const uint32_t* | - |
| dL_dpixels | const float* | - |
| dL_invdepths | const float* | - |
| dL_dmean2D | float3* | - |
| dL_dconic2D | float4* | - |
| dL_dopacity | float* | - |
| dL_dcolors | float* | - |
| dL_dinvdepths | float* | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| void | - |

## 3. 操作逻辑

1. 待补充

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_code_learn/files/submodules/diff-gaussian-rasterization/cuda_rasterizer/3dgs_backward.h\|backward.h]] |
| 所属类 | - |
| 命名空间 | BACKWARD |
| 类型 | declaration |
