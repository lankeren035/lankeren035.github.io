---
title: "preprocess"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.backward.cu.BACKWARD.preprocess"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| P | int | - |
| D | int | - |
| M | int | - |
| means3D | const float3* | - |
| radii | const int* | - |
| shs | const float* | - |
| clamped | const bool* | - |
| opacities | const float* | - |
| scales | const glm::vec3* | - |
| rotations | const glm::vec4* | - |
| scale_modifier | const float | - |
| cov3Ds | const float* | - |
| viewmatrix | const float* | - |
| projmatrix | const float* | - |
| focal_x | const float | - |
| focal_y | float | - |
| tan_fovx | const float | - |
| tan_fovy | float | - |
| campos | const glm::vec3* | - |
| dL_dmean2D | const float3* | - |
| dL_dconic | const float* | - |
| dL_dinvdepth | const float* | - |
| dL_dopacity | float* | - |
| dL_dmean3D | glm::vec3* | - |
| dL_dcolor | float* | - |
| dL_dcov3D | float* | - |
| dL_dsh | float* | - |
| dL_dscale | glm::vec3* | - |
| dL_drot | glm::vec4* | - |
| antialiasing | bool | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| void | - |

## 3. 操作逻辑

1. 待补充

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_code_learn/files/submodules/diff-gaussian-rasterization/cuda_rasterizer/3dgs_backward.cu\|backward.cu]] |
| 所属类 | - |
| 命名空间 | BACKWARD |
| 类型 | namespace_scoped |
