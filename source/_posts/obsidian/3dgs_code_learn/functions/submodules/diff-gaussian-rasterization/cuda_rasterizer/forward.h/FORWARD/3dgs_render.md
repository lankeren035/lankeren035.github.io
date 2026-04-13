---
title: "render"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.forward.h.FORWARD.render"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| grid | const dim3 | - |
| block | dim3 | - |
| ranges | const uint2* | - |
| point_list | const uint32_t* | - |
| W | int | - |
| H | int | - |
| points_xy_image | const float2* | - |
| features | const float* | - |
| conic_opacity | const float4* | - |
| final_T | float* | - |
| n_contrib | uint32_t* | - |
| bg_color | const float* | - |
| out_color | float* | - |
| depths | float* | - |
| depth | float* | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| void | - |

## 3. 操作逻辑

1. 待补充

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_code_learn/files/submodules/diff-gaussian-rasterization/cuda_rasterizer/3dgs_forward.h\|forward.h]] |
| 所属类 | - |
| 命名空间 | FORWARD |
| 类型 | declaration |
