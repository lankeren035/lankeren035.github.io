---
title: "rasterize_gaussians"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.diff_gaussian_rasterization.__init__.py.rasterize_gaussians"
hexo-path:
---

## 1. 输入

| 参数名             | 类型  | 解释                                              |
| --------------- | --- | ----------------------------------------------- |
| means3D         | -   | 高斯中心【p，3】                                       |
| means2D         | -   | 2D坐标展位张量【p,3】，为了让它在反向传播时能**接住屏幕空间 2D 均值相关的梯度**。 |
| sh              | -   | 球谐系数【p，16，3】                                    |
| colors_precomp  | -   | 使用sh在python端计算的高斯颜色（如有）                         |
| opacities       | -   | 不透明度【p，1】                                       |
| scales          | -   | 高斯尺度【p，3】                                       |
| rotations       | -   | 高斯旋转【p，4】                                       |
| cov3Ds_precomp  | -   | 使用缩放和旋转在python端计算的协方差（如有）                       |
| raster_settings | -   | 渲染参数[[3dgs_GaussianRasterizationSettings]]      |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| - | - |

## 3. 操作逻辑

1. 调用python端的torch接口（可自动求导）[[source/_posts/obsidian/3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py/_RasterizeGaussians/3dgs_forward|3dgs_forward]]

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[source/_posts/obsidian/3dgs_code_learn/files/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/3dgs___init__.py|__init__.py]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | - |
