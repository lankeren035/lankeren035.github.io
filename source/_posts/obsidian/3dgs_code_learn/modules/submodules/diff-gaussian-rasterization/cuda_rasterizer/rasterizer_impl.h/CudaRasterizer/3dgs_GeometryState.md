---
title: "GeometryState"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.rasterizer_impl.h.CudaRasterizer.GeometryState"
hexo-path:
---
- `GeometryState` 不是那种“一个高斯一个 struct”的简单布局。它更像一个**视图结构**，里面有很多指针成员
- 用于存渲染过程中“每个高斯自己的几何/外观中间结果”
## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| - | - | - |

## 2. 属性

| 属性名            | 类型        | 来源函数    | 解释                                                             |
| -------------- | --------- | ------- | -------------------------------------------------------------- |
| scan_size      | size_t    | <class> | 做 **prefix sum / scan** 时需要的临时空间大小                             |
| depths         | float*    | <class> | 每个高斯的深度值（后面要按 **tile + depth** 排序）                             |
| scanning_space | char*     | <class> | 给前缀和算法用的 **实际工作区内存地址**                                         |
| clamped        | bool*     | <class> | 记录 **RGB 三个通道是否被 clamp 过**                                     |
| internal_radii | int*      | <class> | 每个高斯投影到屏幕后，在内部表示下的 **半径/覆盖范围**（用于判断会覆盖哪些tile）                  |
| means2D        | float2*   | <class> | 每个高斯投影到图像平面后的 2D 中心位置                                          |
| cov3D          | float*    | <class> | 3D 协方差矩阵的压缩表示（对称矩阵保存6个数）                                       |
| conic_opacity  | float4*   | <class> | 椭圆二次型的 3 个参数与不透明度                                              |
| rgb            | float*    | <class> | 高斯颜色                                                           |
| point_offsets  | uint32_t* | <class> | 做前缀和以后得到的 **每个高斯在复制列表里的起始写入位置**（假设0号高斯覆盖了3个tile，那么1号高斯重位置3开始写） |
| tiles_touched  | uint32_t* | <class> | 每个高斯一共覆盖了多少个 tile                                              |

## 3. 方法

| 方法名 | 解释 |
|---|---|
| [[source/_posts/obsidian/3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h/CudaRasterizer/GeometryState/3dgs_fromChunk\|fromChunk]] | - |
