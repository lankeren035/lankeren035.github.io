---
title: "fusedssimCUDA"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.fused-ssim.ssim.cu.fusedssimCUDA"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| H | int | - |
| W | int | - |
| CH | int | - |
| C1 | float | - |
| C2 | float | - |
| img1 | float* | - |
| img2 | float* | - |
| ssim_map | float* | - |
| dm_dmu1 | float* | - |
| dm_dsigma1_sq | float* | - |
| dm_dsigma12 | float* | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| __global__ void | - |

## 3. 操作逻辑

1. 待补充

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_ssim.cu\|ssim.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
