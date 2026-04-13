---
title: "forward"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.diff_gaussian_rasterization.__init__.py._RasterizeGaussians.forward"
hexo-path:
---

## 1. 输入

| 参数名             | 类型  | 解释                                              |
| --------------- | --- | ----------------------------------------------- |
| ctx             | -   | 上下文对象，用于前向传播和反向传播之间传递信息                         |
| means3D         | -   | 高斯中心【p，3】                                       |
| means2D         | -   | 2D坐标展位张量【p,3】，为了让它在反向传播时能**接住屏幕空间 2D 均值相关的梯度**。 |
| sh              | -   | 球谐系数【p，16，3】                                    |
| colors_precomp  | -   | 使用sh在python端计算的高斯颜色（如有）                         |
| opacities       | -   | 不透明度【p，1】                                       |
| scales          | -   | 高斯尺度【p，3】                                       |
| rotations       | -   | 高斯旋转【p，4】                                       |
| cov3Ds_precomp  | -   | 使用缩放和旋转在python端计算的协方差（如有）                       |
| raster_settings | -   | 渲染参数[[3dgs_GaussianRasterizationSettings]]      |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| - | - |

## 3. 操作逻辑

1. 打包如下参数（args）：
	- 高斯核心参数：
		-  means3D：（P, 3)，每个点的3D坐标   
		- colors_precomp：python预计算的高斯颜色【N, 3】（如有）
		- opacities：（P, 1)，每个点的不透明度，初始0.1
		- scales：协方差缩放
		- rotations：协方差旋转
		- cov3Ds_precomp：python预计算的3D协方差矩阵 【N, 3, 3】（如有）
		-   sh： 球谐系数【p, 16, 3】
	- 渲染参数（raster_settings）：
		- bg ：背景颜色
		- scale_modifier：对所有高斯的尺度做一个统一调节。值大，高斯更“胖”；值小，高斯更“瘦”，代码中默认=1
		- viewmatrix, # 世界坐标系转换到相机坐标系
		- projmatrix：完整投影矩阵，把3D视锥投影到NDC，x 和 y 轴的范围是 \[-1, 1]，z 轴范围通常是 \[0, 1]
		- tanfovx,：水平视场角（半角）的切线值
		- tanfovy：垂直视场角（半角）的切线值
		- image_height：图像高度
		- image_width：图像宽度
		- sh_degree：当前启用的球谐阶数
		- campos：相机中心在世界坐标系中的位置
		- prefiltered,  # 是否启用预过滤
		- antialiasing：是否启用抗锯齿
		- debug：是否启用调试模式
2. 调用C++渲染接口
	- 走声明[[source/_posts/obsidian/3dgs_code_learn/files/submodules/diff-gaussian-rasterization/3dgs_setup.py|setup.py]]
	- 走调用
3. 返回参数：
	- num_rendered：高斯实例数（复制后）
	- color：渲染图【3，H，W】
	- radii：每个高斯在当前视角下的 **屏幕空间近似半径（外接矩形）**【P】
	- geomBuffer：几何缓冲区（里面存了高斯预处理阶段的所有中间结果缓存）[[3dgs_GeometryState]]
	- binningBuffer：分桶排序缓冲区（存索引）[[3dgs_BinningState]]
	- imgBuffer：图像/像素阶段的辅助缓存（存tile范围索引）
	- invdepths： 逆深度图【1, H, W】
## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[source/_posts/obsidian/3dgs_code_learn/files/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/3dgs___init__.py\|__init__.py]] |
| 所属类 | [[3dgs__RasterizeGaussians\|_RasterizeGaussians]] |
| 命名空间 | - |
| 类型 | - |
