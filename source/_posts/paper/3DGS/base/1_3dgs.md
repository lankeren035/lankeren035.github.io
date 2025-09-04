---
title:  "3DGS_捏雪球"

date:  2025-8-30 15:28:00

tags:  [3DGS]

categories:  [3DGS]

comment:  false

toc:  true
---

#

<!--more-->

> - 基于splatting和机器学习的三维重建方法
>   - 无深度学习
>   - 简单的机器学习
>   - 大量的CG知识
>   - 复杂的线性代数
>   - 对GPU的高性能编程

## 0. 问题

|                  问题                  | 回答 |
| :------------------------------------: | :--: |
|            什么是splatting             |      |
|           什么是3D Gaussian            |      |
|            如何进行参数估计            |      |
|        为什么3d gaussian是椭球         |      |
|     协方差矩阵怎么就能控制椭球形状     |      |
| 协方差矩阵怎么就能用旋转和缩放矩阵表达 |      |
|        各向异性和各向同性是什么        |      |
|            为什么引入雅可比            |      |
|     球谐函数怎么就能更好的表达颜色     |      |
|             3dgs怎么就快了             |      |



## 1. Splatting

### 1.1 基本理解

- 一种体渲染方法：从3D物体渲染到2D平面
- NeRF中用的体渲染方法叫Ray-casting，它是被动的
  - 我有一张图片，然后我从每个像素出发去找到影响这个像素的发光粒子，最后确定这个像素的颜色。这个过程中主角是像素
- splatting是主动的
  - 计算出每个发光粒子如何影响像素点，主角是粒子。

### 1.2 什么叫splatting：

- Splat：拟声词，【啪唧一声】
- 想象输入是一些雪球，图片是一面砖墙
- 图像生成过程是向墙面扔雪球的过程
- 每扔一个雪球，墙面上会有扩散痕迹，足迹（footprint）
- 同时会有啪唧一声，由此得名
- 所以这个算法也称为抛雪球算法，或者足迹法
- 翻译成喷溅也很有灵性

### 1.3 Splatting流程

1. 选择【雪球】
2. 抛雪球：从3D投影到2D，得到足迹
3. 加以合成，形成最后的图像



## 2. 选择雪球

### 2.1 为什么使用核（雪球）

- 我们的输入是点云中的一些点，点是没有体积的，他也没有能量，所以我们对他进行一个膨胀
- 需要给点一个核来进行膨胀
- 核可以选择高斯/圆/正方体...

### 2.2 为什么选择3D高斯椭球

- 很好的数学性质：
  - 放射变换后高斯核仍然闭合
  - 3D将为到2D后（沿着某一个轴积分）仍然为高斯
- 定义：
  - 椭球高斯$G( x ) = \frac{ 1 }{ \sqrt{ ( 2 \pi )^ k | \Sigma | } } e^ { - \frac{ 1 }{ 2 } ( x - \mu )^ T \Sigma ^ { - 1 } ( x - \mu ) }$
  - $\Sigma$表示协方差矩阵，半正定，$| \Sigma |$ 是其行列式

### 2.3 3d gaussian为什么是椭球

- 通常椭球（面）：
  $$
  \frac{ x^ 2 }{ a^ 2 }+ \frac{ y^ 2 }{ b^ 2 } + \frac{ z^ 2 }{ c^ 2 } = 1 \\
  Ax^ 2 + By^ 2+ Cz^2 + 2Dxy + 2Exz + 2Fyz = 1
  $$
  
- 3D高斯：
  $$
  G( x ) = \frac{ 1 }{ \sqrt{ ( 2 \pi )^ k | \Sigma | } } e^ { - \frac{ 1 }{ 2 } ( x - \mu )^ T \Sigma ^ { - 1 } ( x - \mu ) }
  $$

  - $\mu$是均值
  - $\Sigma$是协方差矩阵
  - 高斯明显不是上面的椭球那种形式啊？它的值是一个概率，他可能是0.1，0.2...，它为什么是一个椭球呢？

- 首先需要知道协方差矩阵
  - 对于高斯分布，在一维的时候他的形状是由均值核方差来决定的，当维度上升时，方差就变成了协方差矩阵
  - 协方差矩阵：$\Sigma = \begin{bmatrix} \sigma_ { x } ^ 2 \space \sigma_{ xy } \space \sigma_ { xz } \\ \sigma_ { yx } \space \sigma_{ y } ^ 2 \space \sigma_ { yz } \\ \sigma_ { zx } \space \sigma_{ zy } \space \sigma_ { z } ^ 2 \\ \end{bmatrix}$
    - 一个对称矩形，决定高斯分布形状
    - 对角线上元素为x轴/y轴/z轴的方差
    - 反斜对角线上的值为协方差，表示x和y，x和z...的线性相关性

- 对于公式：$G( x, \mu, \Sigma ) = \frac{ 1 }{ \sqrt{ ( 2 \pi )^ k | \Sigma | } } e^ { - \frac{ 1 }{ 2 } ( x - \mu )^ T \Sigma ^ { - 1 } ( x - \mu ) }$

  - $\frac{ 1 }{ \sqrt{ ( 2 \pi )^ k | \Sigma | } } $ 这部分是常数

  - 当$(x - \mu )^ T \Sigma^ { -1 } (x  - \mu )$ 是一个常数时，整个高斯的概率就是一个常数了。因此：

    - 一维的时候：x等于一个常数
      $$
      \begin{align}
      (x - \mu )^ T \Sigma^ { -1 } (x  - \mu ) &= c \\
      \frac{ ( x - \mu )^ 2 }{ \sigma^ 2 } = c
      \end{align}
      $$

    - 二维的时候：椭圆方程
      $$
      \begin{align}
      (x - \mu )^ T \Sigma^ { -1 } (x  - \mu ) &= c \\ 
      ([x,y] - [\mu_ 1, \mu_ 2]) ^T \begin{bmatrix} 
       \sigma_ { x } ^ 2 \space \sigma_{ xy } \\ \sigma_ { yx } \space \sigma_{ y } ^ 2 \\ 
      \end{bmatrix}^ { - 1 } ([x,y] - [\mu_ 1, \mu_ 2]) &= c \\
      \frac{ ( x - \mu_ 1 )^ 2 }{ \sigma_ 1 ^ 2 } + \frac{ ( y - \mu_ 2 )^ 2 }{ \sigma_ 2 ^ 2 } - \frac{ 2 \sigma_{ xy } ( x - \mu_ 1 )( y - \mu_ 2 ) }{ \sigma_ 1 \sigma_ 2 } &= c
      \end{align}
      $$

    - 三维的时候：椭球面方程
      $$
      \begin{align}
      (x - \mu )^ T \Sigma^ { -1 } (x  - \mu ) &= c \\ 
      \frac{ ( x - \mu_ 1 )^ 2 }{ \sigma_ 1 ^ 2 } + \frac{ ( y - \mu_ 2 )^ 2 }{ \sigma_ 2 ^ 2 } + \frac{ ( z - \mu_ 3 )^ 2 }{ \sigma_ 3 ^ 2 } - \frac{ 2 \sigma_{ xy } ( x - \mu_ 1 )( y - \mu_ 2 ) }{ \sigma_ 1 \sigma_ 2 } - \frac{ 2 \sigma_{ xz } ( x - \mu_ 1 )( z - \mu_ 3 ) }{ \sigma_ 1 \sigma_ 3 } - \frac{ 2 \sigma_{ yz } ( y - \mu_ 2 )( z - \mu_ 3 ) }{ \sigma_ 2 \sigma_ 3 } &= c
      \end{align}
      $$
      

  - 由于$G(x , \mu, \Sigma) \in [\times , \times]$，对公式进行变换，可以得到上面的$c \in [\times, \times]$ 

    - $(x - \mu )^ T \Sigma^ { -1 } (x  - \mu ) = c$ 定义了一个椭球面
    - $G(x , \mu, \Sigma) \in [\times,\times]$ 大椭球壳套小椭球壳
    - 实心的椭球

### 2.4 各向同性&各向异性

- 各向同性：
  - 在所有方向具有相同的扩散程度（梯度）
  - 球
  - 3d高斯分布：协方差矩阵是对角矩阵，且对角线元素相等：$\Sigma = \begin{bmatrix} \sigma^ 2 &0 &0 \\ 0 &\sigma^ 2 &0 \\ 0 &0 &\sigma^ 2 \\ \end{bmatrix}$

- 各向异性：
  - 在不同方向具有不同的扩散程度（梯度）
  - 椭球
  - 3d高斯分布：协方差矩阵是对角矩阵：$\Sigma = \begin{bmatrix} \sigma_ { x }^ 2 &\sigma_ { xy } &\sigma_ { xz } \\ \sigma_ { yx } &\sigma_ { y }^ 2 &\sigma_ { yz } \\ \sigma_ { zx } &\sigma_ { zy } &\sigma_ { z }^ 2 \\ \end{bmatrix}$

### 2.5 协方差矩阵是如何控制椭球形状的

- 高斯分布：$\mathbf{x} \sim N ( \mu , \Sigma)$
  - 均值$[\mu_ 1, \mu_2, \mu_ 3]$
  - 协方差矩阵$\begin{bmatrix} \sigma_ { x }^ 2 &\sigma_ { xy } &\sigma_ { xz } \\ \sigma_ { yx } &\sigma_ { y }^ 2 &\sigma_ { yz } \\ \sigma_ { zx } &\sigma_ { zy } &\sigma_ { z }^ 2 \\ \end{bmatrix}$

- 高斯分布的仿射变换：
  $$
  \begin{align}
  \mathbf{ w } &= A \mathbf{ x } + b \\
  \mathbf{ w } &\sim N ( A \mu + b , A \Sigma A^ { T })
  \end{align}
  $$
  
- 标准高斯分布：$\mathbf{x} \sim N ( \vec{ 0 } , I)$
  - 均值[0,0,0]
  - 协方差矩阵$\begin{bmatrix} 1 &0 &0 \\ 0 &1 &0 \\ 0 &0 &1 \\ \end{bmatrix}$

- 任意高斯可以看作是标准高斯通过仿射变换得到
  
  - 球经过仿射变换后变成椭球

### 2.6 协方差矩阵可通过旋转和缩放矩阵表达

- 高斯分布的仿射变换：
  $$
  \begin{align}
  \mathbf{ w } &= A \mathbf{ x } + b \\
  \mathbf{ w } &\sim N ( A \mu + b , A \Sigma A^ { T })
  \end{align}
  $$

  - 根据仿射变换的概念，A本质上就是一个旋转乘以缩放：（仿射变换通常是通过旋转、缩放、平移来完成的）
    $$
    A = RS
    $$

  - 因此：
    $$
    \begin{align}
    \Sigma &= A \cdot I \cdot A^ T \\
    &=R \cdot S \cdot I \cdot ( R \cdot S )^ T \\
    &= R \cdot S \cdot S^ T \cdot  R^ T
    \end{align}
    $$
    

- 我已经知道协方差矩阵了，怎么求$R、S$呢？

  - 通过特征值分解将$\Sigma$分解成$Q \Lambda Q^ T$

  - $\Lambda$是一个对角矩阵，他的对角线元素是他的特征值：$\begin{bmatrix} S_0 &0 &0 \\ 0 &S_1 &0 \\ 0 &0 &S_ 2 \end{bmatrix}$

  - 最后拆成：$\Sigma = Q \Lambda^{ \frac{ 1 }{ 2 }} \Lambda^{ \frac{ 1 }{ 2 }} Q^ T $

    ```python
    import numpy as np
    def computeCov3D(scale, rot, mod=1): #scale是缩放矩阵，rot旋转矩阵，mod是一个缩放
        # 缩放矩阵
        S = np.array(
            [
                [scale[0] * mod, 0, 0],
                [0, scale[1] * mod, 0],
                [0, 0, scale[2] * mod] 
            ]
        )
        R = rot
        M = np.dot(R,S)
        cov3D = np.dot(M,M.T)
    
        return cov3D
    ```

    