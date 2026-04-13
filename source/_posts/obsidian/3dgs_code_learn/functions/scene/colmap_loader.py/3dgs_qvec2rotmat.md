---
title: "qvec2rotmat"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/scene.colmap_loader.py.qvec2rotmat"
hexo-path:
---
- 旋转四元数$q=(w,x,y,z)$->旋转矩阵公式：
	$$
	R =  
	\begin{bmatrix}  
	1 - 2y^2 - 2z^2 & 2xy - 2wz & 2xz + 2wy \\  
	2xy + 2wz & 1 - 2x^2 - 2z^2 & 2yz - 2wx \\  
	2xz - 2wy & 2yz + 2wx & 1 - 2x^2 - 2y^2  
	\end{bmatrix}​​
	$$

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| qvec | - | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| - | - |

## 3. 操作逻辑

1. 待补充

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_colmap_loader.py\|colmap_loader.py]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | - |
