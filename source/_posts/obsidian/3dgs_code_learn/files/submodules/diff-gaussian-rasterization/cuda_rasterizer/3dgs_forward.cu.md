---
title: "forward.cu"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.forward.cu"
hexo-path:
---

## 1. 流程

1. 待补充

## 2. 包含的类

| 类名 | 解释 |
|---|---|
| - | - |

## 3. 包含的函数

| 函数名 | 返回 | 解释 |
|---|---|---|
| [[3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu/3dgs_computeColorFromSH\|computeColorFromSH]] | __device__ glm::vec3 | - |
| [[3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu/3dgs_computeCov2D\|computeCov2D]] | __device__ float3 | - |
| [[3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu/3dgs_computeCov3D\|computeCov3D]] | __device__ void | - |
| [[3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu/3dgs_preprocessCUDA\|preprocessCUDA]] | template<int C> __global__ void | - |
| [[3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu/3dgs_renderCUDA\|renderCUDA]] | template <uint32_t CHANNELS> __global__ void __launch_bounds__(BLOCK_X * BLOCK_Y) | - |
| [[3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu/FORWARD/3dgs_render\|FORWARD::render]] | void | - |
| [[3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu/FORWARD/3dgs_preprocess\|FORWARD::preprocess]] | void | - |
