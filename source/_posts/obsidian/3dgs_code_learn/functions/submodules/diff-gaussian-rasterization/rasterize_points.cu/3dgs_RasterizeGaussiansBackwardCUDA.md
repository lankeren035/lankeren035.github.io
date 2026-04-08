---
title: "RasterizeGaussiansBackwardCUDA"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.rasterize_points.cu.RasterizeGaussiansBackwardCUDA"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| background | const torch::Tensor& | - |
| means3D | const torch::Tensor& | - |
| radii | const torch::Tensor& | - |
| colors | const torch::Tensor& | - |
| opacities | const torch::Tensor& | - |
| scales | const torch::Tensor& | - |
| rotations | const torch::Tensor& | - |
| scale_modifier | const float | - |
| cov3D_precomp | const torch::Tensor& | - |
| viewmatrix | const torch::Tensor& | - |
| projmatrix | const torch::Tensor& | - |
| tan_fovx | const float | - |
| tan_fovy | const float | - |
| dL_dout_color | const torch::Tensor& | - |
| dL_dout_invdepth | const torch::Tensor& | - |
| sh | const torch::Tensor& | - |
| degree | const int | - |
| campos | const torch::Tensor& | - |
| geomBuffer | const torch::Tensor& | - |
| R | const int | - |
| binningBuffer | const torch::Tensor& | - |
| imageBuffer | const torch::Tensor& | - |
| antialiasing | const bool | - |
| debug | const bool | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> | - |

## 3. 操作逻辑

1. 待补充

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_code_learn/files/submodules/diff-gaussian-rasterization/3dgs_rasterize_points.cu\|rasterize_points.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
