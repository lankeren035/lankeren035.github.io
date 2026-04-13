---
title: "forward"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.rasterizer.h.CudaRasterizer.Rasterizer.forward"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| geometryBuffer | std::function<char* (size_t)> | - |
| binningBuffer | std::function<char* (size_t)> | - |
| imageBuffer | std::function<char* (size_t)> | - |
| P | const int | - |
| D | int | - |
| M | int | - |
| background | const float* | - |
| width | const int | - |
| height | int | - |
| means3D | const float* | - |
| shs | const float* | - |
| colors_precomp | const float* | - |
| opacities | const float* | - |
| scales | const float* | - |
| scale_modifier | const float | - |
| rotations | const float* | - |
| cov3D_precomp | const float* | - |
| viewmatrix | const float* | - |
| projmatrix | const float* | - |
| cam_pos | const float* | - |
| tan_fovx | const float | - |
| tan_fovy | float | - |
| prefiltered | const bool | - |
| out_color | float* | - |
| depth | float* | - |
| antialiasing | bool | - |
| radii | int* | - |
| debug | bool | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| static int | - |

## 3. 操作逻辑

1. 待补充

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_code_learn/files/submodules/diff-gaussian-rasterization/cuda_rasterizer/3dgs_rasterizer.h\|rasterizer.h]] |
| 所属类 | [[3dgs_code_learn/modules/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer.h/CudaRasterizer/3dgs_Rasterizer\|Rasterizer]] |
| 命名空间 | CudaRasterizer |
| 类型 | declaration |
