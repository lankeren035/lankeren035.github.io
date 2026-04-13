---
title: "ImageState"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.rasterizer_impl.h.CudaRasterizer.ImageState"
hexo-path:
---
- 存“每个像素 / tile 在渲染阶段的中间结果”
- 里面放的是“这个 tile 对应哪段高斯列表”、“这个像素累积了多少贡献”、“当前累积了多少 alpha”。
## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| - | - | - |

## 2. 属性

| 属性名         | 类型        | 来源函数    | 解释                                              |
| ----------- | --------- | ------- | ----------------------------------------------- |
| ranges      | uint2*    | <class> | 对于每个像素（或 tile），它在排序后高斯列表中对应的区间范围 `[begin, end)` |
| n_contrib   | uint32_t* | <class> | 每个像素当前累计了多少个高斯贡献                                |
| accum_alpha | float*    | <class> | 每个像素目前已经累计到的 alpha（不透明度），这个值用于早停                |

## 3. 方法

| 方法名 | 解释 |
|---|---|
| [[source/_posts/obsidian/3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h/CudaRasterizer/ImageState/3dgs_fromChunk\|fromChunk]] | - |
