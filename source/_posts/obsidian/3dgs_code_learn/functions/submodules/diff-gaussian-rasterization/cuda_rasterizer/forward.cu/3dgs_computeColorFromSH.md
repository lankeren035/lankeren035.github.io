---
title: "computeColorFromSH"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.forward.cu.computeColorFromSH"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| idx | int | - |
| deg | int | - |
| max_coeffs | int | - |
| means | const glm::vec3* | - |
| campos | glm::vec3 | - |
| shs | const float* | - |
| clamped | bool* | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| __device__ glm::vec3 | - |

## 3. 操作逻辑

1. 待补充

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_code_learn/files/submodules/diff-gaussian-rasterization/cuda_rasterizer/3dgs_forward.cu\|forward.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
