---
title: "in_frustum"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.auxiliary.h.in_frustum"
hexo-path:
---

## 1. 输入

| 参数名         | 类型           | 解释                                                       |
| ----------- | ------------ | -------------------------------------------------------- |
| idx         | int          | 高斯编号                                                     |
| orig_points | const float* | （P, 3)，每个点的3D坐标                                          |
| viewmatrix  | const float* | 世界坐标系转换到相机坐标系                                            |
| projmatrix  | const float* | 完整投影矩阵，把3D视锥投影到NDC，x 和 y 轴的范围是 \[-1, 1]，z 轴范围通常是 \[0, 1] |
| prefiltered | bool         | 是否启用预过滤                                                  |
| p_view      | float3&      | 当前高斯的视空间坐标（用于输出）                                         |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| __forceinline__ __device__ bool | - |

## 3. 操作逻辑

1. 将当前高斯坐标（x,y,z）乘以转换矩阵，得到NDC空间坐标：
	$$\begin{bmatrix}
	\frac{2n}{r-l} & 0 & 0 & 0 \\
	0 & \frac{2n}{t-b} & 0 & 0 \\
	0 & 0 & \frac{f}{f-n} & -\frac{nf}{f-n} \\
	0 & 0 & 1 & 0
	\end{bmatrix} \times 
	\begin{bmatrix}
	x \\
	y \\
	z  \\
	1 
	\end{bmatrix} =
	\begin{bmatrix}
	\frac{2nx}{r-l} \\
	\frac{2ny}{t-b} \\
	\frac{fz-nf}{f-n} \\
	z 
	\end{bmatrix} 
	$$
2. 上式得到的结果，最后一位为z，在齐次坐标中点的最后一位应该是1，因此需要做透视除法得到结果：
	$$
	\begin{bmatrix}
	\frac{2nx}{(r-l)z} \\
	\frac{2ny}{(t-b)z} \\
	\frac{fz-nf}{(f-n)z} \\
	1 
	\end{bmatrix} $$
3. 判断当前高斯是否在视锥空间内：
	- 视锥空间判断：z< 0.2f 的点排除
	- NDC空间判断：$x \in [-1.3,1.3], y \in [-1.3, 1.3]$ （该条实际计算时注释了）（）
4. 该函数最终筛选出距离镜头太近的高斯，将高斯转换到视锥空间（没做透视除法）

**思考**：NDC空间判断为什么选1.3，正常是1吧 ？
- 高斯有半径：即便中心在 [-1,1] 外，只要半径够大，足迹还是会覆盖到屏幕边缘的像素。如果你严格以 1.0 裁掉，会出现边缘闪烁（中心刚越界就整颗被砍）
## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_auxiliary.h\|auxiliary.h]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | definition |
