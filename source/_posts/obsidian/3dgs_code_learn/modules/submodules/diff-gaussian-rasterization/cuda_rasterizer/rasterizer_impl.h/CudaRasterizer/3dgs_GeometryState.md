---
title: "GeometryState"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.rasterizer_impl.h.CudaRasterizer.GeometryState"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| - | - | - |

## 2. 属性

| 属性名 | 类型 | 来源函数 | 解释 |
|---|---|---|---|
| scan_size | size_t | <class> | - |
| depths | float* | <class> | - |
| scanning_space | char* | <class> | - |
| clamped | bool* | <class> | - |
| internal_radii | int* | <class> | - |
| means2D | float2* | <class> | - |
| cov3D | float* | <class> | - |
| conic_opacity | float4* | <class> | - |
| rgb | float* | <class> | - |
| point_offsets | uint32_t* | <class> | - |
| tiles_touched | uint32_t* | <class> | - |

## 3. 方法

| 方法名 | 解释 |
|---|---|
| [[source/_posts/obsidian/3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h/CudaRasterizer/GeometryState/3dgs_fromChunk\|fromChunk]] | - |
