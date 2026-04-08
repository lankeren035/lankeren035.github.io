---
title: "preprocessCUDA"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.forward.cu.preprocessCUDA"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| P | int | - |
| D | int | - |
| M | int | - |
| orig_points | const float* | - |
| scales | const glm::vec3* | - |
| scale_modifier | const float | - |
| rotations | const glm::vec4* | - |
| opacities | const float* | - |
| shs | const float* | - |
| clamped | bool* | - |
| cov3D_precomp | const float* | - |
| colors_precomp | const float* | - |
| viewmatrix | const float* | - |
| projmatrix | const float* | - |
| cam_pos | const glm::vec3* | - |
| W | const int | - |
| H | int | - |
| tan_fovx | const float | - |
| tan_fovy | float | - |
| focal_x | const float | - |
| focal_y | float | - |
| radii | int* | - |
| points_xy_image | float2* | - |
| depths | float* | - |
| cov3Ds | float* | - |
| rgb | float* | - |
| conic_opacity | float4* | - |
| grid | const dim3 | - |
| tiles_touched | uint32_t* | - |
| prefiltered | bool | - |
| antialiasing | bool | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| template<int C> __global__ void | - |

## 3. 操作逻辑

1. 待补充

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_code_learn/files/submodules/diff-gaussian-rasterization/cuda_rasterizer/3dgs_forward.cu\|forward.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
