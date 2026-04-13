---
title: "required"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.rasterizer_impl.h.CudaRasterizer.required"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| P | size_t | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| template<typename T> size_t | - |

## 3. 操作逻辑

1. 根据调用者不同，调用它里面的fromChunk函数：
	- GeometryState调用：[[source/_posts/obsidian/3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu/CudaRasterizer/GeometryState/3dgs_fromChunk|GeometryState::fromChunk]]
	- BinningState调用：[[source/_posts/obsidian/3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu/CudaRasterizer/BinningState/3dgs_fromChunk|BinningState::fromChunk]]
	- ImageState调用：[[source/_posts/obsidian/3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu/CudaRasterizer/ImageState/3dgs_fromChunk|ImageState::fromChunk]]

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_rasterizer_impl.h|rasterizer_impl.h]] |
| 所属类 | - |
| 命名空间 | CudaRasterizer |
| 类型 | definition |
