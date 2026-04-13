---
title: "ndc2Pix"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.auxiliary.h.ndc2Pix"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| v | float | - |
| S | int | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| __forceinline__ __device__ float | - |

## 3. 操作逻辑

```c
__forceinline__ __device__ float ndc2Pix(float v, int S)  //像素坐标的原点在左上角像素的中心，有个0.5的偏移
{
    return ((v + 1.0) * S - 1.0) * 0.5;
}
```

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_auxiliary.h\|auxiliary.h]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
