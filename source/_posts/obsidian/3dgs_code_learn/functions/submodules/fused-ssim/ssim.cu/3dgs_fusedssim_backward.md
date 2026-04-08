---
title: "fusedssim_backward"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.fused-ssim.ssim.cu.fusedssim_backward"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| C1 | float | - |
| C2 | float | - |
| img1 | torch::Tensor & | - |
| img2 | torch::Tensor & | - |
| dL_dmap | torch::Tensor & | - |
| dm_dmu1 | torch::Tensor & | - |
| dm_dsigma1_sq | torch::Tensor & | - |
| dm_dsigma12 | torch::Tensor & | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| torch::Tensor | - |

## 3. 操作逻辑

1. 待补充

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_code_learn/files/submodules/fused-ssim/3dgs_ssim.cu\|ssim.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
