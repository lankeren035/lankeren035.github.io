---
title: "preprocessCUDA"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.forward.cu.preprocessCUDA"
hexo-path:
---
- 将每个3D高斯变成可光栅化的2D表示
	- 它在不在当前相机视锥里
	- 它投到屏幕上的中心在哪
	- 它投影后的 2D 协方差是多少
	- 它的 2D 椭圆覆盖半径是多少
	- 它会打到多少个 tile
	- 它渲染时要用的颜色、深度、conic 参数是什么
## 1. 输入

| 参数名             | 类型               | 解释                                                          |     |
| --------------- | ---------------- | ----------------------------------------------------------- | --- |
| P               | int              | 点数                                                          |     |
| D               | int              | 当前启用的球谐阶数                                                   |     |
| M               | int              | 球谐系数总个数（3阶为16个）                                             |     |
| orig_points     | const float*     | （P, 3)，每个点的3D坐标                                             |     |
| scales          | const glm::vec3* | 协方差缩放                                                       |     |
| scale_modifier  | const float      | 对所有高斯的尺度做一个统一调节。值大，高斯更“胖”；值小，高斯更“瘦”，代码中默认=1                 |     |
| rotations       | const glm::vec4* | 协方差旋转                                                       |     |
| opacities       | const float*     | （P, 1)，每个点的不透明度，初始0.1                                       |     |
| shs             | const float*     | 球谐系数【p, 16, 3】                                              |     |
| clamped         | bool*            | 来自[[3dgs_GeometryState]]，记录 RGB 三个通道是否被 clamp 过             |     |
| cov3D_precomp   | const float*     | python预计算的3D协方差矩阵 【N, 3, 3】（如有）                             |     |
| colors_precomp  | const float*     | python预计算的高斯颜色【N, 3】（如有）                                    |     |
| viewmatrix      | const float*     | 世界坐标系转换到相机坐标系                                               |     |
| projmatrix      | const float*     | 完整投影矩阵，把3D视锥投影到NDC，x 和 y 轴的范围是 \[-1, 1]，z 轴范围通常是 \[0, 1]    |     |
| cam_pos         | const glm::vec3* | 相机中心在世界坐标系中的位置                                              |     |
| W               | const int        | 图像宽度                                                        |     |
| H               | int              | 图像高度                                                        |     |
| tan_fovx        | const float      | 水平视场角（半角）的切线值                                               |     |
| tan_fovy        | float            | 垂直视场角（半角）的切线值                                               |     |
| focal_x         | const float      | x轴焦距                                                        |     |
| focal_y         | float            | y轴焦距                                                        |     |
| radii           | int*             | 高斯屏幕半径（外接矩形）                                                |     |
| points_xy_image | float2*          | 来自[[3dgs_GeometryState]]，每个高斯投影到图像平面后的 2D 中心位置              |     |
| depths          | float*           | 来自[[3dgs_GeometryState]]，每个高斯的深度值（后面要按 **tile + depth** 排序） |     |
| cov3Ds          | float*           | 来自[[3dgs_GeometryState]]，3D 协方差矩阵的压缩表示（对称矩阵保存6个数）           |     |
| rgb             | float*           | 来自[[3dgs_GeometryState]]，高斯颜色                               |     |
| conic_opacity   | float4*          | 来自[[3dgs_GeometryState]]，椭圆二次型的 3 个参数与不透明度                  |     |
| grid            | const dim3       | 将图片划分成多个tiles的实例                                            |     |
| tiles_touched   | uint32_t*        | 来自[[3dgs_GeometryState]]，每个高斯一共覆盖了多少个 tile                  |     |
| prefiltered     | bool             | 是否启用预过滤                                                     |     |
| antialiasing    | bool             | 是否启用抗锯齿                                                     |     |
## 2. 输出

| 返回类型 | 解释 |
|---|---|
| template<int C> __global__ void | - |

## 3. 操作逻辑

1. 取当前线程的线程号，对应的线程号用于处理对应的高斯
2. 视锥裁剪+得到视空间坐标
	- 判断当前高斯中心是否在视锥范围内[[3dgs_in_frustum]]（实际做的是排除离镜头太近的点 0.2f）
	- 对合法高斯，将高斯中心变换到NDC空间得到p_proj（最后一步是做透视除法，保证齐次坐标最后一位是1，实际除以（z+0.0000001））：
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
		\end{bmatrix} =
		\begin{bmatrix}
		\frac{2nx}{(r-l)z} \\
		\frac{2ny}{(t-b)z} \\
		\frac{fz-nf}{(f-n)z} \\
		1 
		\end{bmatrix} 
		$$
3. 计算协方差
	1) 计算3D协方差：有python预计算则直接用，否则cuda计算[[source/_posts/obsidian/3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu/3dgs_computeCov3D|3dgs_computeCov3D]]
	2) 从3D协方差投影到2D协方差[[3dgs_computeCov2D]]                               
	3) 给2D协方差对角线增加0.3，让高斯变胖一点，让矩阵更稳定、更容易求逆
4. 如果启用抗锯齿，则计算一个缩放因子，修正卷积后的 opacity，在协方差被膨胀后，**尽量保持卷积前后积分量一致**。直觉上，协方差变大了，斑点会铺得更宽， 如果 α 不变，总能量会失真， 所以要把 α 按面积比例缩一下。##todo
5. 计算2D高斯的二次型表示
	1) 原始2D协方差：
		$$\Sigma = \begin{bmatrix}
		xx & xy \\
		yx & yy
		\end{bmatrix}$$
	2) 计算其行列式：
		$$det(\Sigma) = xx \cdot yy - (xy)^2$$
	3) 2D协方差求逆：
		$$\Sigma^{-1}= det^{-1}\begin{bmatrix}
		yy & -xy \\
		-yx & xx
		\end{bmatrix}$$
	4) 存储$\Sigma^{-1}$，需要用三个数，存储到：conic（二次型系数，渲染时使用）
6. 根据 2D 协方差特征值估计屏幕覆盖半径
	- 计算特征值（特征值对应椭圆主轴方向上的方差）：
		$$\begin{align}
		\lambda_{1}= \frac{xx+yy}{2} + max(0.1 \space , \space (\frac{xx+yy}{2})^2-det)\\
		\lambda_{2}= \frac{xx+yy}{2} - max(0.1 \space , \space (\frac{xx+yy}{2})^2-det)
		\end{align}$$
	- 根据特征值计算$3\sigma$ 长轴半径：(这里会做一个ceil保证整数，因为后面覆盖范围按像素 / tile 算，得取整数半径)
		$$3 \times \sqrt{max(\lambda_{1} \space,\space \lambda_2)}$$
7. 视口变换：将高斯中心从NDC坐标\[-1,1]映射到屏幕坐标（左上角原点）[[3dgs_ndc2Pix]]
8. 画出高斯的$3\sigma$椭圆的外接矩形，求出该矩形的像素坐标范围（左上角、右下角），并映射到 tile 坐标系  ， 这里的grid是渲染时的tile grid（按照16\*16为一个tile，每个每个tile为最小单位组成的grid）[[3dgs_getRect]] ##todo
9. 根据外接矩大小，若太小=0，则抛弃该高斯
10. 根据sh计算高斯颜色[[source/_posts/obsidian/3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu/3dgs_computeColorFromSH|3dgs_computeColorFromSH]] ##todo
11. 获得每个高斯的深度：depths\[i] = z（没做透视除法的ndc空间）
12. 获得每个高斯的像素半径：长轴$3\sigma$
13. 统计该高斯覆盖的tiled数量（根据外接矩的左上右下坐标计算）
## 4. 信息

| 字段   | 内容                  |              |
| ---- | ------------------- | ------------ |
| 所属文件 | [[3dgs_forward.cu]] | forward.cu]] |
| 所属类  | -                   |              |
| 命名空间 | -                   |              |
| 类型   | definition          |              |
