---
title: "fromChunk"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.rasterizer_impl.cu.CudaRasterizer.GeometryState.fromChunk"
hexo-path:
---
- 把 `chunk` 指向的一整块连续原始内存，按 128 字节对齐的规则，依次切成 GeometryState 需要的各个数组，并把这些数组首地址填进 `geom` 里返回；同时 `chunk` 被一路往后推进到“已分配完毕”的末尾位置。
## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| chunk | char*& | - |
| P | size_t | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| CudaRasterizer::GeometryState | - |

## 3. 操作逻辑

1. 定义一个空的[[3dgs_GeometryState|GeometryState]]结构体
2. 计算内存分配：
	- 分配depths：从 `chunk` 当前指向的位置开始，按 128 字节对齐，切出一段能容纳 `P` 个 `depths` 元素的空间，把首地址赋给 `geom.depths`，然后把 `chunk` 挪到这段空间后面（这里的“元素类型”不是在这一句里显式写出来的，而是由 `geom.depths` 的类型决定的。如果 `geom.depths` 是 `float*`，那就是切出 `P * sizeof(float)` 字节。）
	- 分配clamped：【p，3】，每个高斯 RGB 三个通道里，哪些分量在 SH 转颜色时被 clamp 了（前向里如果对颜色做了 clamp，反向传播时梯度处理会受影响。  所以需要把这个信息缓存下来，后面 backward 才知道哪些通道在前向被截断过。）
	- 分配internal_radii：【p】，它和外部传进来的 `radii` 是同类信息，只不过这是 `GeometryState` 自己内部备用的一份缓冲。
	- 分配means2D：【p，2】，屏幕空间高斯中心坐标，通常类型会是 `float2*`
	- 分配cov3D：【p，3，3】，对称矩阵因此协方差6个数
	- 分配conic_opacity：【p，4】，2D 椭圆/圆锥曲线相关参数与不透明度
	- 分配rgb：【p，3】，高斯颜色
	- 分配tiles_touched：【p】，每个原始高斯覆盖了多少个 tile
	- 计算如果要对长度为P的数组做前缀和，需要的空间大小，然后分配
	- 分配point_offsets：【p】，前缀和结果数组，每个位置存的是截至当前高斯的累计 tile 覆盖数

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_rasterizer_impl.cu\|rasterizer_impl.cu]] |
| 所属类 | - |
| 命名空间 | CudaRasterizer::GeometryState |
| 类型 | namespace_scoped |
