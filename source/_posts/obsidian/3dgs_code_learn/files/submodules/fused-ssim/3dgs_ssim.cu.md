---
title: "ssim.cu"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.fused-ssim.ssim.cu"
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
| [[3dgs_code_learn/functions/submodules/fused-ssim/ssim.cu/3dgs_get_pix_value\|get_pix_value]] | __device__ float | - |
| [[3dgs_code_learn/functions/submodules/fused-ssim/ssim.cu/3dgs_load_into_shared\|load_into_shared]] | __device__ void | - |
| [[3dgs_code_learn/functions/submodules/fused-ssim/ssim.cu/3dgs_multiply_shared_mem\|multiply_shared_mem]] | __device__ void | - |
| [[3dgs_code_learn/functions/submodules/fused-ssim/ssim.cu/3dgs_do_sq\|do_sq]] | __device__ inline float | - |
| [[3dgs_code_learn/functions/submodules/fused-ssim/ssim.cu/3dgs_flush_conv_scratch\|flush_conv_scratch]] | __device__ void | - |
| [[3dgs_code_learn/functions/submodules/fused-ssim/ssim.cu/3dgs_do_separable_conv_x\|do_separable_conv_x]] | __device__ void | - |
| [[3dgs_code_learn/functions/submodules/fused-ssim/ssim.cu/3dgs_do_separable_conv_y\|do_separable_conv_y]] | __device__ float | - |
| [[3dgs_code_learn/functions/submodules/fused-ssim/ssim.cu/3dgs_fusedssimCUDA\|fusedssimCUDA]] | __global__ void | - |
| [[3dgs_code_learn/functions/submodules/fused-ssim/ssim.cu/3dgs_fusedssim_backwardCUDA\|fusedssim_backwardCUDA]] | __global__ void | - |
| [[3dgs_code_learn/functions/submodules/fused-ssim/ssim.cu/3dgs_fusedssim\|fusedssim]] | std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor> | - |
| [[3dgs_code_learn/functions/submodules/fused-ssim/ssim.cu/3dgs_fusedssim_backward\|fusedssim_backward]] | torch::Tensor | - |
