---
title: "computeCov3D"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.backward.cu.computeCov3D"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| idx | int | - |
| scale | const glm::vec3 | - |
| mod | float | - |
| rot | const glm::vec4 | - |
| dL_dcov3Ds | const float* | - |
| dL_dscales | glm::vec3* | - |
| dL_drots | glm::vec4* | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| __device__ void | - |

## 3. 操作逻辑

1. 待补充

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_backward.cu\|backward.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
