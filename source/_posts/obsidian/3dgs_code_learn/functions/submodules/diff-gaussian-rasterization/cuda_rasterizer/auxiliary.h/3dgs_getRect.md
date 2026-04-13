---
title: "getRect"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.auxiliary.h.getRect"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| p | const float2 | - |
| ext_rect | int2 | - |
| rect_min | uint2& | - |
| rect_max | uint2& | - |
| grid | dim3 | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| __forceinline__ __device__ void | - |

## 3. 操作逻辑

```c
__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)

{

    rect_min = {

        min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))), //左边界的tile索引

        min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y))) //上边界的tile索引

    };

    rect_max = {

        min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))), //右边界的tile索引

        min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y))) //下边界的tile索引

    };

}
```

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_auxiliary.h\|auxiliary.h]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
