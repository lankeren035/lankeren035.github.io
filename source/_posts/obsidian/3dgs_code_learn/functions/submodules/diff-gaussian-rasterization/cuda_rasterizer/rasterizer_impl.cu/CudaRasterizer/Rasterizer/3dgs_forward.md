---
title: "forward"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.rasterizer_impl.cu.CudaRasterizer.Rasterizer.forward"
hexo-path:
---

## 1. 输入

| 参数名            | 类型                            | 解释                                                                                                                                                                 |
| -------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| geometryBuffer | std::function<char* (size_t)> | 这里本质上传入的不是缓冲区，而是一个函数，用于创建缓冲区，该函数在外层已经传递了geometryBuffer进去，因此该函数用于[[3dgs_GeometryState]]的创建，具体调用的是[[3dgs_resizeFunctional]]函数，该函数将geometryBuffer这个tensor resize到目标大小 |
| binningBuffer  | std::function<char* (size_t)> | 同上，用于创建[[3dgs_BinningState]]                                                                                                                                       |
| imageBuffer    | std::function<char* (size_t)> | 同上，用于创建[[3dgs_ImageState]]                                                                                                                                         |
| P              | const int                     | 点数                                                                                                                                                                 |
| D              | int                           | 当前启用的球谐阶数                                                                                                                                                          |
| M              | int                           | 球谐系数总个数（3阶为16个）                                                                                                                                                    |
| background     | const float*                  | 背景颜色                                                                                                                                                               |
| width          | const int                     | 图像宽度                                                                                                                                                               |
| height         | int                           | 图像高度                                                                                                                                                               |
| means3D        | const float*                  | （P, 3)，每个点的3D坐标                                                                                                                                                    |
| shs            | const float*                  | 球谐系数【p, 16, 3】                                                                                                                                                     |
| colors_precomp | const float*                  | python预计算的高斯颜色【N, 3】（如有）                                                                                                                                           |
| opacities      | const float*                  | （P, 1)，每个点的不透明度，初始0.1                                                                                                                                              |
| scales         | const float*                  | 协方差缩放                                                                                                                                                              |
| scale_modifier | const float                   | 对所有高斯的尺度做一个统一调节。值大，高斯更“胖”；值小，高斯更“瘦”，代码中默认=1                                                                                                                        |
| rotations      | const float*                  | 协方差旋转                                                                                                                                                              |
| cov3D_precomp  | const float*                  | python预计算的3D协方差矩阵 【N, 3, 3】（如有）                                                                                                                                    |
| viewmatrix     | const float*                  | 世界坐标系转换到相机坐标系                                                                                                                                                      |
| projmatrix     | const float*                  | 完整投影矩阵，把3D视锥投影到NDC，x 和 y 轴的范围是 \[-1, 1]，z 轴范围通常是 \[0, 1]                                                                                                           |
| cam_pos        | const float*                  | 相机中心在世界坐标系中的位置                                                                                                                                                     |
| tan_fovx       | const float                   | 水平视场角（半角）的切线值                                                                                                                                                      |
| tan_fovy       | float                         | 垂直视场角（半角）的切线值                                                                                                                                                      |
| prefiltered    | const bool                    | 是否启用预过滤                                                                                                                                                            |
| out_color      | float*                        | 输出渲染图                                                                                                                                                              |
| depth          | float*                        | 输出逆深度图                                                                                                                                                             |
| antialiasing   | bool                          | 是否启用抗锯齿                                                                                                                                                            |
| radii          | int*                          | 高斯屏幕半径（外接矩形）                                                                                                                                                       |
| debug          | bool                          | 是否启用调试模式                                                                                                                                                           |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| int | - |

## 3. 操作逻辑

1. 计算焦距
2. 创建三个缓存：
	1. 做一次dry run计算[[3dgs_GeometryState|GeometryState]]这一整套几何阶段数组，在处理 `P` 个原始高斯时，总共需要多少字节。(这个过程传入空指针，计算指针偏移，不会分配空间，只会算出来depth指针指向a cov3D指针指向a+128\*P得出最终指针偏移也就是GeometryState结构体所需的空间大小由于是按照空指针计算的中间这些指针都是无用的)[[3dgs_required]]
		- 另外两个缓存类似：[[3dgs_BinningState|binningBuffer]]、[[3dgs_ImageState|imageBuffer]]
	2. 调用创建函数[[3dgs_resizeFunctional|resizeFunctional]]，创建[[3dgs_GeometryState|geometryBuffe]]
		- 另外两个缓存类似
	3. 给GemoetryState结构体的各个成员变量分配显存,每个变量有p个元素
		- 另外两个缓存类似
3. 划分tiles和block
	- 将图片划分为多个tiles
	- 将每个tile作为一个cuda的block，使用16\*16线程，一个线程对应一个像素
4. 预处理高斯[[source/_posts/obsidian/3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu/FORWARD/3dgs_preprocess|3dgs_preprocess]]
## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_rasterizer_impl.cu|rasterizer_impl.cu]] |
| 所属类 | - |
| 命名空间 | CudaRasterizer::Rasterizer |
| 类型 | namespace_scoped |
