---
title: "backward"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.rasterizer_impl.cu.CudaRasterizer.Rasterizer.backward"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| P | const int | - |
| D | int | - |
| M | int | - |
| R | int | - |
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
| campos | const float* | - |
| tan_fovx | const float | - |
| tan_fovy | float | - |
| radii | const int* | - |
| geom_buffer | char* | - |
| binning_buffer | char* | - |
| img_buffer | char* | - |
| dL_dpix | const float* | - |
| dL_invdepths | const float* | - |
| dL_dmean2D | float* | - |
| dL_dconic | float* | - |
| dL_dopacity | float* | - |
| dL_dcolor | float* | - |
| dL_dinvdepth | float* | - |
| dL_dmean3D | float* | - |
| dL_dcov3D | float* | - |
| dL_dsh | float* | - |
| dL_dscale | float* | - |
| dL_drot | float* | - |
| antialiasing | bool | - |
| debug | bool | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| void | - |

## 3. 操作逻辑

1. 待补充

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_code_learn/files/submodules/diff-gaussian-rasterization/cuda_rasterizer/3dgs_rasterizer_impl.cu\|rasterizer_impl.cu]] |
| 所属类 | - |
| 命名空间 | CudaRasterizer::Rasterizer |
| 类型 | namespace_scoped |
