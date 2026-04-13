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

| 参数名        | 类型               | 解释                                              |
| ---------- | ---------------- | ----------------------------------------------- |
| idx        | int              | 高斯索引/线程索引                                       |
| deg        | int              | 当前启用的球谐阶数                                       |
| max_coeffs | int              | 球谐系数总个数（3阶为16个）                                 |
| means      | const glm::vec3* | （P, 3)，每个点的3D坐标                                 |
| campos     | glm::vec3        | 相机中心在世界坐标系中的位置                                  |
| shs        | const float*     | 球谐系数【p, 16, 3】                                  |
| clamped    | bool*            | 来自[[3dgs_GeometryState]]，记录 RGB 三个通道是否被 clamp 过 |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| __device__ glm::vec3 | - |

## 3. 操作逻辑

1. 待补充

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_forward.cu\|forward.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
