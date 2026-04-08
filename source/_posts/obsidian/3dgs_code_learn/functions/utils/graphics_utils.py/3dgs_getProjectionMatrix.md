---
title: "getProjectionMatrix"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/utils.graphics_utils.py.getProjectionMatrix"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| znear | - | - |
| zfar | - | - |
| fovX | - | - |
| fovY | - | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| - | - |

## 3. 操作逻辑

1. 本函数需要求一个矩阵，将3D视锥空间转换为正交投影空间。他需要做如下两个步骤：
	1) 通过矩阵$M_{perspective \rightarrow ortho}$将透视投影的视锥空间 转换成 正交投影空间
		$$
			M_{persp\rightarrow ortho}=  
			\begin{bmatrix}  
			n & 0 & 0 & 0 \\  
			0 & n & 0 & 0 \\  
			0 & 0 & n+f & -nf \\  
			0 & 0 & 1 & 0  
			\end{bmatrix}
        $$
	2) 通过$M_{ortho}$执行正交投影：将该空间平移到坐标系中心，然后缩放到固定大小
		- 这里需要注意的是，有的方法是做正交投影是将视锥中心放到坐标系中心，映射范围是缩放至$[-1,1]^ 3$的正方体，而本文是xy压缩到[-1,1]，z是压缩到[0,1]，因此本文的$M_{ortho}^{[0,1]}$ 跟别的常见的矩阵不一样	![](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/paper/3DGS/base/3dgs/12.png)
		  $$
				M_{ortho}=  
				\begin{bmatrix}  
				\frac{2}{r-l} & 0 & 0 & -\frac{r+l}{r-l} \\  
				0 & \frac{2}{t-b} & 0 & -\frac{t+b}{t-b} \\  
				0 & 0 & \frac{1}{f-n} & -\frac{n}{f-n} \\  
				0 & 0 & 0 & 1  
				\end{bmatrix}
			$$
	3) 将上面两个矩阵组合得到视锥空间到NDC空间的映射矩阵：
		$$
			P_{theory}=  M_{ortho} M_{persp \rightarrow ortho} =
			\begin{bmatrix}  
			\frac{2n}{r-l} & 0 & -\frac{r+l}{r-l} & 0 \\  
			0 & \frac{2n}{t-b} & -\frac{t+b}{t-b} & 0 \\  
			0 & 0 & \frac{f}{f-n} & -\frac{nf}{f-n} \\  
			0 & 0 & 1 & 0  
			\end{bmatrix}
		 $$
	 4) 代码中计算的映射矩阵形式：
		 $$
		     P_{code}=
			\begin{bmatrix}
			\frac{1}{\tan(fovX/2)} & 0 & \frac{r+l}{r-l} & 0 \\
			0 & \frac{1}{\tan(fovY/2)} & \frac{t+b}{t-b} & 0 \\
			0 & 0 & \frac{f}{f-n} & -\frac{nf}{f-n} \\
			0 & 0 & 1 & 0
			\end{bmatrix}=
			\begin{bmatrix}
			\frac{2n}{r-l} & 0 & \frac{r+l}{r-l} & 0 \\
			0 & \frac{2n}{t-b} & \frac{t+b}{t-b} & 0 \\
			0 & 0 & \frac{f}{f-n} & -\frac{nf}{f-n} \\
			0 & 0 & 1 & 0
			\end{bmatrix}=
			\begin{bmatrix}
			\frac{2n}{r-l} & 0 & 0 & 0 \\
			0 & \frac{2n}{t-b} & 0 & 0 \\
			0 & 0 & \frac{f}{f-n} & -\frac{nf}{f-n} \\
			0 & 0 & 1 & 0
			\end{bmatrix}
           $$
           ![62](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/obsidian/3dgs/2.png)
           - 可以发现这里第三列
## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_graphics_utils.py\|graphics_utils.py]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | - |
