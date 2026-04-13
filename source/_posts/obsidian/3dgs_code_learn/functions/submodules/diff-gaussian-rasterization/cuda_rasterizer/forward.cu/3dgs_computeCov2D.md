---
title: "computeCov2D"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.forward.cu.computeCov2D"
hexo-path:
---
- 根据3D协方差，计算出投影到屏幕上的2D协方差
## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| mean | const float3& | - |
| focal_x | float | - |
| focal_y | float | - |
| tan_fovx | float | - |
| tan_fovy | float | - |
| cov3D | const float* | - |
| viewmatrix | const float* | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| __device__ float3 | - |

## 3. 操作逻辑

1. 首先将视锥空间的点进行限制，将x，y坐标clamp到\[-1.3fov，1.3fov]范围
2. 梳理投影过程：
	1) 3D协方差 -> 相机坐标系3D协方差（放射变换，直接乘以W2C）
		$$R_{w2c} \cdot cov3D \cdot R_{w2c}^T$$
	2) 相机坐标系3D协方差 -> 正交空间3D协方差（该过程将透视投影转为正交投影，非仿射变换，要用[[2_3dgs#4.2.2 基于雅可比矩阵的投影变换|雅可比]]）
		$$J \cdot R_{w2c} \cdot cov3D \cdot R_{w2c}^{T}\cdot J^ T$$
	3) 正交空间3D协方差 -> 2D协方差（正交投影，去掉z）

**注意**：将协方差投影到2D的时候没有压缩到NDC盒子，直接用的图像尺度，因此后面无需视口变换。而高斯位置/均值 是处理到了NDC空间的，因此要做视口变换
**思考**：为什么要clamp到1.3范围？
- 因为计算雅可比矩阵的时候靠近视锥边缘的点会出现：x/z或y/z过大，导致数值不稳定，后面求逆得到的二次型也会失真，甚至崩溃
## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_forward.cu|forward.cu]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
