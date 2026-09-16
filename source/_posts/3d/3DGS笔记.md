---
title: 3DGS笔记
date: 2026-09-10
tags:
  - 3DGS
categories:
  - 3DGS
comment: true
toc: true
published: true
permalink: 3d
hexo-path:
---
#
<!--more-->

# 0. 计算机图形学
## 0.1 坐标变换

- 通过**左乘**一个4\*4矩阵表达对3D空间点的**平移、旋转，缩放**
### 0.1.1 平移
- 平移矩阵相当于单位矩阵加上一个平移向量：

$$
\begin{bmatrix} 1 &0 &0 &x_ e \\\\ 0 &1 &0 &y_e \\\\ 0 &0 &1 &z_ e \\\\ 0 &0 &0 &1 \end{bmatrix}
$$
- 举例：将点[1,0,0,1]沿着x周平移1个单位：$M(x_e = 1)*[1,0,0,1] ^T = [2,0,0,1] ^T$
## 0.1.2 旋转

https://blog.csdn.net/csxiaoshui/article/details/65446125

## 0.1.3 缩放
- 单位矩阵在各个轴乘以缩放系数
$$
\begin{bmatrix}
\frac{ 2 }{ r - l } &0 &0 &0 \\\\
0 &\frac{ 2 }{ t - b } &0 &0 \\\\
0 &0 &\frac{ 2 }{ n - f } &0 \\\\
0 &0 &0 &1 \\\\
\end{bmatrix}
$$
    - 视锥体长方体 [l,r]×[b,t]×[n,f][l,r]×[b,t]×[n,f] 线性映射到 NDC 立方体的纯缩放部分


## 0.2 视图变换

- 拍照的过程：
  1. 模型变换：先把场景搭好（得到世界坐标系）
  2. 视图变换：把世界坐标变成「以相机为参考」（得到相机坐标系）
  3. 投影变换：把相机坐标压到裁剪体/屏幕
- 已知相机在世界坐标系里「站在哪、朝哪看、头朝哪」，怎么把所有 3D 点变到「以相机为原点、朝 -z 看、头顶是 +y」的标准相机坐标系？
### 0.2.1 视图变换流程

#### 1. 定义相机坐标系
- 如何才能确定一个相机的摆放？
  1) 首先，相机的位置很重要，用位置向量$\vec{ e }=(x_ e, y_ e, z_ e)$
  2) 其次，往哪拍也很重要，用:
    - look-at向量（相机往哪看）：$\hat g$
    - 向上向量（相机头顶的朝向）：$\hat t$
- 默认相机在原点，$\hat g$是-z轴，$\hat t$是y轴，右手坐标系


  ![](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/paper/3DGS/base/3dgs/16.png)
#### 2. world to camera
- 核心目的：将整个场景内的物体和相机做同样的一个变换，变换之后相机成了坐标系中心。这就是视图变换
1) 将相机**平移**到原点，平移矩阵：$T _ { view }  = \begin{bmatrix} 1 &0 &0 &-x_ e \\\\ 0 &1 &0 &-y_e \\\\ 0 &0 &1 &-z_ e \\\\ 0 &0 &0 &1 \end{bmatrix}$
2)  **旋转**相机，以对齐三轴：将$\hat{ g }$ 旋转到-z轴，将$\hat t$ 旋转到y轴，$\hat g \times \hat t$旋转到x轴
    - 直接将$\hat g$旋转到 -z轴不太好操作，旋转矩阵不太好找，可以逆向考虑
        1. 假设我要将x轴旋转到(a,b,c)这个单位向量：$\begin{bmatrix} a &none &none &none \\\\ b &none &none &none \\\\ c &none &none &none \\\\ 0 &0 &0 &1 \end{bmatrix} [1,0,0,0]^T = [a,b,c,0]^T$
        2. 同理，如果你要将y轴，z轴旋转到（a,b,c）只需要第二列第三列是abc就行，那么如果你要同时将世界坐标系的三轴旋转到相机坐标系的三轴只需要三列分别为xyz轴的目标单位向量即可
        3. 因此将-z轴根据$R_ { view } ^ { - 1 }$转到$\hat g$，然后由于旋转矩阵的逆矩阵只需要转置就行，求个逆就可以找到旋转矩阵。 $R_ { view } ^ { - 1 } = \begin{bmatrix} x_{ \hat g \times \hat t } &x_ t &x_ { - g } &0 \\\\ y_ { \hat g \times \hat t } &y_t  &y_ { -g } &0 \\\\ z_ { \hat g \times \hat t } &z_ t &z_{ -g } &0 \\\\ 0 &0 &0 &1 \end{bmatrix}$

3) 得到$T_ { view }$和$R_ { view }$后，将相机和其他物体都做一个这样的变换，保持相对关系，就完成了视图变换
## 0.3 投影变换

- 投影矩阵不是直接把 3D 点变成 2D 像素，而是先把**相机坐标**变成**裁剪空间坐标**。真正变成 2D 图像是在后面的“透视除法 + 视口映射 + 光栅化”阶段 
    - 为什么要先变成立方体（而不是直接二维）？
        - 我们最终要在屏幕上画 2D 像素，但在画之前必须知道谁挡住谁、哪些部分该被裁掉、属性怎么做“透视正确插值”。这些都离不开 z/w，所以业界标准是：世界 → 相机 → 投影（得到裁剪坐标）→ 透视除法（NDC 立方体）→ 屏幕。 
        - **统一裁剪**：无论正交还是透视，先把视见体变成统一的标准盒子（或在齐次裁剪空间的盒状约束），硬件用同一种裁剪逻辑就能处理，效率高。
         - **保留深度**：在产出 2D 像素前，需要 z 来做隐藏面消除、透明度排序、阴影/雾等效果。
         - **数值/规范化**：把范围归一到\[-1,1]（或 z∈[0,1]）能简化插值、测试和精度控制。
- 有两种投影方式：
  - 正交投影
  - 透视投影
    - 有远小近大

### 0.3.1  正交投影

- 正交投影的作用是把一个长方体视见体映射到 NDC 立方体。
#### 1. 流程

  - 假设你要对空间中的一个立方体：$[l,r]\times[b,t]\times[-f,-n]$ ，里面的所有内容进行正交投影

  - 先将立方体的中心移到原点，然后将xyz分别缩放到[-1,1]
    $$
    M_ { ortho } = 
    \begin{bmatrix}
    \frac{ 2 }{ r - l } &0 &0 &0 \\\\
    0 &\frac{ 2 }{ t - b } &0 &0 \\\\
    0 &0 &\frac{ 2 }{ n - f } &0 \\\\
    0 &0 &0 &1 \\\\
    \end{bmatrix}
    \begin{bmatrix}
    1 &0 &0 &- \frac{ r+l }{2} \\\\
    0 &1 &0 &- \frac{ t+b }{2} \\\\
    0 &0 &1 &- \frac{ n+f }{2} \\\\
    0 &0 &0 &1
    \end{bmatrix} \\\\
    =  
    \begin{bmatrix}  
    \frac{2}{r-l}&0&0&-\frac{r+l}{r-l}\\\\  
    0&\frac{2}{t-b}&0&-\frac{t+b}{t-b}\\\\  
    0&0&-\frac{2}{f-n}&-\frac{f+n}{f-n}\\\\  
    0&0&0&1  
    \end{bmatrix}
    $$
    

### 0.3.2 透视投影

- 使用最广泛，远小近大，平行线就不再平行了
![](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/paper/3DGS/base/3dgs/13.png)

- 透视投影通过一个视锥观察物体，假设近平面是n，远平面式f，如上图（下），透视投影要做的就是将线的右端点投影到左端点，透视投影跟正交投影的区别就是远平面要大些。

#### 1. 流程：

- 假设对远平面进行压缩，把他**压缩**成跟近平面一样大，（图下从左到右的过程），然后将远平面的点通过**正交投影**就可以投影到近平面。相当于将透视投影拆分成两个过程：
  1. 压缩平面，将远平面到近平面这里所有的平面都压缩成跟近平面一样大
  2. 对压缩后得到的立方体做正交投影
##### 1）压缩平面

1. 几个规定假设：
  - 近平面压缩后不变（例如顶点压缩后还是原来的顶点，中点压缩后还是原来的中点）
  - 远平面压缩后z值不会变化
  - 远平面的中心点压缩后不变

![](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/paper/3DGS/base/3dgs/17.png)

2. 对于视锥上的一条线，他在某平面（不一定是远平面）上的点为a (x，y，z)，他在近平面上的点为b（x'，y'，z'）；a点压缩后（视锥压缩成立方体），会被压缩成（x‘，y’，z''），且根据三角形相似可知：$y' = \frac{ -n }{ z} y$   $x' =  \frac{ -n }{ z } x$ (z<0, n>0)。对于某平面（不一定是远平面）任意一点$(x, y,z ,1 )^ T$ 会被压缩成$( \frac{ -nx }{ z }, \frac{ -ny }{ z } , unknown , 1 )$ ，最后一步除以-z就是透视除法，透视除法不是直接变成 2D。透视除法后得到的是 NDC 坐标:
  $$
  \left (  \begin{matrix} n&0 &0 &0 \\\\
  0 &n &0 &0 \\\\ ? &? &? &? \\\\
  0 &0 &-1 &0
  \end{matrix} \right) \times \left (  \begin{matrix} x \\\\
  y \\\\ z \\\\
  1
  \end{matrix} \right) = \left (  \begin{matrix} nx \\\\
  ny \\\\ ? \\\\
  -z
  \end{matrix} \right) =\left (  \begin{matrix} \frac{ -nx }{ z } \\\\
  \frac{ -ny }{ z } \\\\ ? \\\\
  1
  \end{matrix} \right)
  $$

      - 所以$M_ { persp \rightarrow ortho } = \left (  \begin{matrix} n&0 &0 &0 \\\\    0 &n &0 &0 \\\\ ? &? &? &? \\\\  0 &0 &-1 &0   \end{matrix} \right)$

3. 对于近平面上的点，压缩后不变：
  $$
  \left (  \begin{matrix} n&0 &0 &0 \\\\
  0 &n &0 &0 \\\\ 0 &0 &A &B \\\\
  0 &0 &-1 &0
  \end{matrix} \right) \times \left (  \begin{matrix} x' \\\\
  y' \\\\ -n \\\\
  1
  \end{matrix} \right) = \left (  \begin{matrix} nx' \\\\
  ny' \\\\ -n^ 2 \\\\
  n
  \end{matrix} \right) =\left (  \begin{matrix} \frac{ nx' }{ n } \\\\
  \frac{ ny' }{ n } \\\\ -n \\\\
  1
  \end{matrix} \right) \\\\ , -An+B=-n^2
  $$
  
4. 远平面的中心点压缩后不变：$-Af + B = -f^ 2$

5. 根据$\left \{ \begin{matrix}  -An+B=-n^2  \\\\ -Af + B = -f^ 2 \\\\\end{matrix}\right.$ 得：
  $$
  M_ { persp \rightarrow ortho } = \left (  \begin{matrix} n&0 &0 &0 \\\\
  0 &n &0 &0 \\\\ 0 &0 &n+f &nf \\\\
  0 &0 &-1 &0
  \end{matrix} \right)  
  $$

6. 最后$M_ { persp } = M_ { ortho } M _ { persp \rightarrow ortho}=\begin{bmatrix} \dfrac{2n}{r-l} & 0 & \dfrac{r+l}{r-l} & 0 \\\\[6pt] 0 & \dfrac{2n}{t-b} & \dfrac{t+b}{t-b} & 0 \\\\[6pt] 0 & 0 & -\dfrac{f+n}{f-n} & -\dfrac{2fn}{f-n} \\\\[6pt] 0 & 0 & -1 & 0 \end{bmatrix}$ 

- **近大远小的来源**：对于相机坐标系空间的点$x,y,z$，经过投影矩阵得到裁剪空间坐标，再经过透视除法得到：点$x_{ndc},y_{ndc},z_{ndc}$ ，

$$
x_{ndc}
=
-\frac{2n}{r-l}\frac{x}{z}
-\frac{r+l}{r-l}
$$
$$
y_{ndc}
=
-\frac{2n}{t-b}\frac{y}{z}
-\frac{t+b}{t-b}
$$
$$
z_{ndc}
=
\frac{f+n}{f-n}
+
\frac{2fn}{(f-n)z}
$$

    - 因为相机前方 \(z<0\)，所以这个式子会把：$z=-n$ 映射到：$z_{ndc}=-1$ 。把：$z=-f$ 映射到：$z_{ndc}=1$ ，透视投影的核心是：$x_{ndc},y_{ndc}\propto \frac{1}{z}$ 由于相机前方 \(z<0\)，距离越远，\(|z|\) 越大，$\frac{1}{|z|}$越小，所以物体投影到屏幕上越小。

---

## 0.4 光栅化

- 目标：把 NDC 中的坐标映射到屏幕像素坐标。
- 注意：这里不是简单丢掉 \(z\)。\(x,y\) 用来确定屏幕位置，\(z\) 仍然用于深度测试。
### 0.4.1 流程
1. 首先，不管z轴，只看x，y轴，先将[-1,1]范围的x，y映射到[0, width] x [0, height]的平面：
2. 缩放加平移（之前的原点在立方体中心，变换后在左上角）

  $$
  M_ { viewport } = \left( \begin{matrix} 
  \frac{width }{ 2 } &0 &0 &\frac{ width }{ 2 } \\\\
  0 &-\frac{ height }{ 2 } &0 &\frac{ height }{ 2 } \\\\
  0 &0 &1 &0 \\\\
  0 &0 &0 &1
  \end{matrix}  \right)
  $$

  

3. 然后判断三角形是否覆盖某个像素。如果覆盖，就计算这个像素的颜色、深度等属性。

4. 多个三角形覆盖同一个像素时，根据深度 \(z\) 判断谁在前面。

---
# 1. NeRF
- NeRF是什么
- NeRF的输入输出是啥
- 怎么把NeRF的输出转化为一张图片
- Loss怎么计算
## 1.1 NeRF是什么
- NeRF是：一种3D表征，类似mesh，点云； 他使用神经网络（MLP）来隐式存储3D信息
    - **显式**的3D信息：**有明确的x,y,z值**（mesh, voxel体素,点云）。
        - 比如mesh会使用一个矩阵$\begin{bmatrix} 1,2,3 \\\\ 2,3,4 \\\\3,4,5 \\\\ 4,5,6 \end{bmatrix}$ 表示有四个顶点，与这四个顶点的xyz坐标。使用$\begin{bmatrix} 1,2,3 \\\\ 1,2,4 \end{bmatrix}$ 表示有两个表面（三角形）顶点1，2，3之间有连接，顶点1，2，4之间有连接。
    - **隐式**的3D信息：无明确的x,y,z值，只能输出指定角度的2D图片。
- NeRF不具有泛化能力，需要针对每个场景进行训练，来一次训一次，不是feed forward模型。
## 1.2 NeRF模型结构
| 问题  | 回答                         | 解释                                                             |
| :-- | -------------------------- | :------------------------------------------------------------- |
| 模型  | 8层MLP                      |                                                                |
| 输入  | 5D向量，$(x,y,z,\theta,\phi)$ | 这是**粒子**的空间位姿 , $xyz$是粒子的空间坐标。<br>$\theta, \phi$表示观察方向的俯仰角、方位角 |
| 输出  | 4D向量，（密度，颜色）               | 这是**粒子**对应的颜色以及密度<br>颜色包括RGB                                   |

## 1.3 粒子假设
### 1.3.1 相机模型
1. **真实场景成像过程**：我们看到东西的时候他会通过**多个反射源**去打光，然后物体和物体之间也会**折射和反射**，这些光最后都会打进我们的眼睛。
    ![](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/paper/3DGS/base/3dgs/1.png)
2. 相机模型：对上述过程进行建模，连接3D世界与2D图片，其中涉及到几个关键的坐标系：
    -  世界坐标系
    - 相机坐标系
    - 归一化相机坐标系（物理成像平面）（CCD）
    - 像素坐标系
       ![](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/paper/3DGS/base/3dgs/2.png)
3. **3D建模（3D重建）**：通过图片去推测光源的位置和强度，以及物体的几何性质（比如他的材质）。这样操作**建模难度巨大**。因此在NeRF中就引入了一个概念：**体渲染** 。
### 1.3.2 体渲染
- 渲染技术的一个分支，在CG领域中，它是为了解决非刚体（云、烟、果冻）（刚体通常有较大密度）的渲染。我们通常将非刚体抽象成一团飘忽不定的**粒子群**。对于这些非刚体，他在成像的时候是光线在穿过的时候会光子和粒子发生一些碰撞。光子和粒子在发生作用的过程中会有四个过程：
    1. 吸收：光子被粒子吸收了
    2. 放射：粒子本身也会发光
    3. 外射光：其他例子向我们反射过来的光
    4. 内射光：我们对其他物体折射的光
- NeRF假设：
    1. 物体是一团自发光的粒子
    2. 粒子有颜色和密度
    3. 外射光和内射光抵消
    4. 多个粒子被渲染成指定角度的图片
      ![](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/paper/3DGS/base/3dgs/8.png)
### 1.3.3 NeRF的输入输出
- 输入：将物体进行稀疏表示的**单个粒子**的位姿
- 输出：该粒子的密度和颜色
- 思考：
    - 模型看上去输入的还是一张图片，输出的也应该是一张图片，我们准备的训练集里面的图片在哪呢？
    - 怎么得到这些粒子？
    - 多少个粒子？这些例子怎么批量输入？
    - 这些粒子是怎么渲染成新的图片的 ？
### 1.3.4 粒子的采集
1. 对于空间中的某一个发光粒子：
    - 空间坐标：**（x,y,z）**
    - 发射的光线通过相机模型成为图片上的像素坐标 **(u,v)**  （从相机出发链接到这个像素点的射线刚好穿过这个粒子）
    - 粒子颜色即为像素颜色
    - [[## 0.2 视图变换|视图变换]]：（u,v）与（x,y,z）的转换公式：相机坐标=相机内参x转换矩阵x世界坐标，该过程包括： [[## 0.2 视图变换|视图变换]] + [[## 0.3 投影变换|投影变换]] + [[#0.4 光栅化|光栅化]] ，呈现形式是一个W2C矩阵
        $$    \begin{bmatrix} u \\\\ v \\\\ 1 \end{bmatrix} = \begin{bmatrix} f_x \space \space 0 \space \space c_x \space \space 0 \\\\ 0 \space \space  f_y \space \space c_y \space \space  0 \\\\ 0 \space \space  0 \space \space  1 \space \space  0 \end{bmatrix}_ { 3 \times 4 } \begin{bmatrix} R \space \space T \\\\ 0 \space \space 1 \end{bmatrix}_ { 4 \times 4} \begin{bmatrix} x_w \\\\ y_w \\\\ z_w \\\\ 1 \end{bmatrix}_ { 4 \times 1}
        $$
        - 其中，转换矩阵为：视图变换矩阵\*透视投影矩阵得到**归一化相机空间**（不是NDC空间，他的xy范围不是[-1,1]，他是焦距为1时的成像平面坐标） ; 内参矩阵为光栅化的缩放与平移
            - 内参fx：当物理焦距f（相机与成像平面的距离）=1（相机常用单位mm）时取成像平面，相机常用像素尺寸$d_x$ 表示一个像素在传感器上的物理大小表示一个pixel代表多大的物理举例，内参$f_x = \frac{f}{d_x}$ ，f=1时，fx表示，一个物理距离表示几个像素（x方向）。而归一化相机空间正是以这个物理距离为单位的，因此通过焦距可以做到物理空间与像素空间距离的转换。
            - 内参cx：表示相机光轴穿过图像平面的那个点，在像素坐标系里的 x 坐标。理想情况下是图像中心。采用单位是像素坐标。
            - 具体内参外参的获取可以参考 [[3DGS笔记#3. COLMAP与SfM]] 
2. 从粒子到像素：已知相机位置A，图片上的像素点B（u,v），我们可以找到一条**射线**。像素点B的颜色可以看作射线上无数个发光点的“和”
    - 一个像素点对应一条射线：$r( t ) = o + td$ 。  $o$ 为射线原点， $d$ 为方向， $t$ 为距离
         ![](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/paper/3DGS/base/3dgs/3.png)
 3. 由像素点P(u,v)反推射线
     ![](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/paper/3DGS/base/3dgs/4.png)
     1) 定义：
         - 像素平面坐标系：$（u,v）$ 
         - 物理成像平面坐标系：$（x_n , y_n）$
         - 相机坐标系：$（x_c , y_c , z_c）$
         - 世界坐标系$（x_w , y_w , z_w）$ 
     2) 像素平面 -> 物理成像平面
         - $(x_n, y_n) = ((u - \frac{w}{2}), -(v- \frac{h}{2}))$ 
     3) 物理成像平面 ->相机坐标系：
         - $( x_c, y_c, z_c)=( x_n, y_n, -f)$ 
         - 归一化：$( x_c, y_c, z_c)=( \frac{x_ c}{ f }, \frac{y_ c }{ f }, -1)$ 
     4) 相机坐标系 ->世界坐标系：
         - $( x_w, y_w, z_w)=W_{c2w} \cdot ( x_c, y_c, z_c)$ 
     5) 根据相机位置（数据集中已知相机的的世界坐标），任意选择一个像素根据上述过程得到该像素的世界坐标。有了**相机坐标**，**像素点坐标**，你就知道射线了。
    - 代码样例
```python
import torch
import numpy as np

# 根据像素点获取射线
# H,W为图片高宽
# K为内参矩阵， 包含fx,fy,cx,cy
# c2w是camera_to_world矩阵
def get_rays(H,W,K,c2w):
    # 生成网格
    '''
    这里调用的时候是先W后H，生成的网格是W行，H列，而我们通常习惯是将图片分成H行，W列的像素，因此这里转置后得到的
    i =
        [[0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3]]
    j = 
        [[0, 0, 0, 0],
        [1, 1, 1, 1],
        [2, 2, 2, 2]]
    看起来i每一列同号，i应该是列号，但是创建网格是WH网格，是反过来的，因此i是行号

    修正：转置之后 i 是列号，j 是行号。
    '''
    i,j = torch.meshgrid(
        torch.linspace(0,W-1,W), #从0到w-1，一共w个数
        torch.linspace(0,H-1,H),
        indexing='ij'
    )
    i,j = i.t(),j.t()
    
    # 像素平面 -> 成像平面 -> 相机坐标系
    # 三个矩阵stack得到【H,W,3】矩阵，z轴坐标-1
    # 得到相机坐标系中的方向向量，因为相机坐标系中相机位置是原点，因此像素坐标就是方向向量
    dirs = torch.stack(
        #K[0][0]是f焦距， K[0][2]是w/2，K[1][2]是H/2，K[1][1]是焦距
        [(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1
    )
    
    # 将方向向量应用c2w的旋转部分，将相机坐标系旋转到世界坐标系，但是原点不平移，因为平移不会改变向量方向
    rays_d = dirs @ c2w[:3, :3].T

    # 射线的起点就是相机中心的 世界坐标，也就是c2w的平移部分
    rays_o = c2w[:3,-1].expand(rays_d.shape)  #expand将其扩展成跟rays_d一样的shape，每个像素点都给一个起点

    return rays_o, rays_d
    
    
rays_o, rays_d = get_rays(H,W,K,c2w)

# H*W个点，每个点的坐标两个数表示
i, j = torch.meshgrid(
    torch.arange(H),
    torch.arange(W),
    indexing='ij'
)
coords = torch.stack([i, j], -1).reshape([-1,2])

# 从中随机选择N_rand个点
select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
select_coords = coords[select_inds].long()

# 查询随机选取的那些坐标上的像素点对应的射线
rays_o = rays_o[select_coords[:,0], select_coords[:,1]]
rays_d = rays_d[select_coords[:,0], select_coords[:,1]]

batch_rays = torch.stack([rays_o, rays_d], 0) #2,N_rand,3

# gt像素值
target_s = target[select_coords[:,0], select_coords[:,1]]

```
4. 理论上，由于射线无限长，t可以取0到正无穷，且可以连续。实际上t是取连续的，而且范围怎么选择呢？
    - 通常设置near=2，far=6，在near和far之间均匀采样64个点
```python
N_sample = 64
t_vals = torch.linspace(0., 1., steps = N_sample)
# 在合法空间采样出64个z，也就是64个粒子
z_vals = near*(1-t_vals) + far* t_vals

# 添加扰动，
if perturb > 0:
    t = (
        # 0，1，2，3，。。。，64
        # 每条射线都进行采样
        # 每条射线radom出来的是0-1的随机数
        # 除以64表示： arrange完成分段，rand完成每一段内取随机数
        torch.arange(N_samples, device=rays_o.device)
        + torch.rand(N_rays, N_samples, device=rays_o.device)
    ) / N_samples
    
    z_vals = near + (far - near) * t
pts = rays_o[..., None, :] + rays_d[..., None,:] * z_vals[...,:,None] #[N_rays, N_samples, 3]
```

5. 训练时，通常一张图片取1024个像素，得到1024条射线，每条射线采样64个粒子，将粒子坐标输入模型[1024\*64,3]
6. 推理时，输入一个粒子的坐标，输出该粒子的颜色和密度

## 1.4 网络结构

- 8层全连接，首先输入粒子坐标，半路再次输入坐标，后半路输出密度$\sigma$ ，密度主要由坐标决定。后半路输入视角（射线方向），最后输出粒子颜色。
![](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/paper/3DGS/base/3dgs/5.png)

### 1.4.1 模型输入-位置编码
- 通常网络输入位置信息如果只输入一个3D的数值，建模结果会有细节丢失，缺乏高频信息。因此需要引入位置编码（位置编码将一个数字编码到不同频率）：
    $$  r(p) = ( sin(2^0 \pi p), cos(2^0 \pi p), ... , sin(2^{L-1} \pi p), cos(2^{L-1} \pi p) )
  $$
    - p为一个标量，x/y/z，需要归一化到[-1,1]
    - $L$ 表示使用多少组不同频率进行位置编码。模型对于粒子坐标取$L=10$，因此一个数字编码成 2\*10 = 20个数字
    - 位置信息为：3\*2\*10 = 60D
    - 模型对于视角（实际上视角使用世界坐标系下的射线方向向量表达）取$L=4$ ，因此视角信息为：3\*2\*4 = 24D
    - 代码中在加上原始坐标值：$r(\mathbf{x})$ 是 63D；$r(\mathbf{d})$ 是 27D
- 为什么这里位置的L设置的比视角的L大？
    - 对于一个粒子来说他的空间坐标的重要性大于他的视角，所以说在模型结构里输出密度的时候还没加入方向信息，说明粒子的**密度只跟它自己本身的绝对空间坐标有关系**，跟视角无关；但是**颜色在各个角度是不一样的**
- 同一条射线上粒子方向是一样的，为什么每个粒子还要编码方向？因为每次采样会采样多条射线1024
### 1.4.2 模型输出
- $\sigma$：查询位置这里有多大概率存在可见物质，即不透明程度。
- Color：查询粒子的颜色。如果这里存在可见物质，它朝观察方向 d 发出的辐射颜色。
    - 这些粒子是怎么渲染成新的图片的？对于图片中每一个像素，计算该像素对应的光线和粒子将这些粒子**通过公式累加**得到该像素最终颜色
#### 1）连续粒子求和
1. 假设有粒子A和B，A在B的前面，如果A异常明亮，那么B的光就不会显示了，粒子之间有遮挡关系，但是实际上一个像素的颜色等于这条射线上粒子颜色的混合。怎么混合呢？用概率密度求期望。假设有AB两个点，那么最终颜色等于：
    $$
    Color(A)*T(A)*\sigma(A) + Color(B)*T(B)*\sigma(B)
    $$
    - Color是如果光线在这里终止，看到的颜色。
    - $T$表示在该粒子之前，光线没有被阻挡的概率，如果这个概率等于0，说明光线无法到达这个粒子，这个粒子的颜色就不起作用了，他贡献的颜色就是0。
    - $\sigma$ 表示光线撞击该粒子（光线被粒子阻挡）的概率密度（光被吸收的概率）到达该粒子的位置之后，光线在单位长度内终止的概率密度，也叫体密度或消光系数（表示当前这个粒子挡光的能力）。
2. 总结为连续情况，像素颜色：    $$
  \begin{align} \hat { C ( s ) } &= \int _0 ^ {+ \infty} T ( s ) \sigma( s ) C ( s ) ds \\\\ T(s) &= e ^ {- \int_ 0 ^ {s}  \sigma( t ) dt} \end{align}
  $$
3.  $T(s)$的推导：对于s位置后面的点$s+ds$，它不被遮挡的概率是：点s不被遮挡且$ds$这一段也都不被遮挡，由于$ds$很小，$ds$这一段被遮挡的概率都视作$\sigma(s)$     $$
    \begin{aligned} 
    T(s + ds ) &= T( s )[ 1 - \sigma ( s ) ds] \\\\
    T(s + ds ) &= T ( s ) - T ( s ) \sigma( s ) ds \\\\
    T ( s + ds ) - T ( s ) &= -T( s ) \sigma( s ) ds \\\\
    dT( s ) &= -T( s ) \sigma( s ) ds \\\\
    \frac{ d T ( s  )}{T ( s )} &= - \sigma( s ) ds \\\\
    \int_0 ^t \frac{dT ( s )}{T( s )} &= \int_0 ^ t - \sigma( s ) \\\\
    \int_0 ^t \frac{1}{T( s )} dT ( s ) &= \int_0 ^ t - \sigma( s ) \\\\
    \text{ ln } T ( s ) |_0 ^t &= \int_0 ^ t - \sigma( s ) \\\\
    \text{ ln } T ( t ) - \text{ ln } T ( 0 ) &= \int_0 ^ t - \sigma( s ) \\\\
    \text{ ln } T ( t )  &= \int_0 ^ t - \sigma( s ) \\\\
    T(t) &= e ^ {- \int_ 0 ^ {t}  \sigma( s ) ds}
    \end{aligned}
    $$
#### 2）离散粒子求和
$$
  \begin{aligned}
  \hat C( r ) &= \sum_{ i = 1} ^ N T_ i ( 1 - e^ {-\sigma_ i  \delta_ i} ) c_ i \\\\
  T_i &= e^ { - \sum_{ j = 1 } ^{ i - 1 } \sigma_ j \delta_ j } 
  \end{aligned}
$$
1. 离散假设：
    - 将光线[0,s]划分为N个等间距区间$[ t_ n \rightarrow t_{ n + 1 }]$ 
    - 间隔长度为$\delta_ n$
    - 假设区间内密度$\sigma_ n$和颜色$C_ n$固定
2. 公式推导：
    - 将连续情况下的积分变成N段的求和：$\hat C = \sum_ { n = 0 } ^ N I ( t_ n \rightarrow t_ {n + 1 })$ 。对于其中的某一段$$
  \begin{aligned}
  I( t_ n \rightarrow t_ { n + 1}) &= \int_{ t_ n } ^ { t_ { n + 1 } } T ( t ) \sigma_n C_n dt \\\\
  &= \sigma_ n C_ n \int _{ t_ n } ^ { t_ { n+ 1 } } T ( t ) dt \\\\
  &= \sigma_ n C_ n \int _{ t_ n } ^ { t_ { n+ 1 } } e^ { - \int_ 0 ^ t \sigma ( s ) ds } dt \\\\
  &= \sigma_ n C_ n \int _{ t_ n } ^ { t_ { n+ 1 } } e^ { - \int_ 0 ^ { t_ n } \sigma ( s ) ds } e^ { - \int_ { t_ n} ^ t \sigma ( s ) ds } dt \\\\
  &= \sigma_ n C_ n T ( 0 \rightarrow t_ n ) \int _{ t_ n } ^ { t_ { n+ 1 } } e^ { - \int_ { t_ n} ^ t \sigma ( s ) ds } dt \\\\
  &= \sigma_ n C_ n T ( 0 \rightarrow t_ n ) \int _{ t_ n } ^ { t_ { n+ 1 } } e^ { - \int_ { t_ n} ^ t \sigma_ n ds } dt (这一步 \sigma 跟s无关) \\\\
  &= \sigma_ n C_ n T ( 0 \rightarrow t_ n ) \int _{ t_ n } ^ { t_ { n+ 1 } } e^ { - \sigma_ n ( t - t_ n ) } dt \\\\
  &= \sigma_ n C_ n T ( 0 \rightarrow t_ n ) [- \frac{1}{ \sigma_ n } e^ { - \sigma_ n ( t - t_ n ) } |_ { t _ n } ^ { t_ n + 1 } ] \\\\
  &= C_ n T ( 0 \rightarrow t_ n )  ( 1 - e ^ { - \sigma_ n \delta_ n} ) \\\\
  &= C_ n T( t_ n )  ( 1 - e ^ { - \sigma_ n \delta_ n} ) \\\\
  &= C_ n  e^ { - \int _ 0 ^ { t_ n} \sigma ( s ) ds }  ( 1 - e ^ { - \sigma_ n \delta_ n} ) \\\\
  &= C_ n  e^ { - \sum _ {i = 0 } ^ { n - 1 } \sigma_ i \delta_ i }  ( 1 - e ^ { - \sigma_ n \delta_ n} ) \\\\
  
  \end{aligned}
  $$
      - 令$\alpha_ n = 1 - e ^ { - \sigma_ n \delta_ n }$最终：    $$
    \begin{aligned}
    \hat C &= \sum_ { n = 0 } ^ N I ( t_ n \rightarrow t_ {n + 1 }) \\\\
    &= \sum_ { n = 0 } ^ N C_ n  e^ { - \sum _ {i = 0 } ^ { n - 1 } \sigma_ i \delta_ i }  ( 1 - e ^ { - \sigma_ n \delta_ n} ) \\\\
    &= \sum_ { n = 0 } ^ N C_ n \alpha_ n ( 1 - \alpha_ 0 ) ( 1 - \alpha_ 1 )...( 1 - \alpha_ { n - 1 } )
    \end{aligned}
    $$
3. 最终颜色的计算形式就是： $C = C_1 \alpha_1+C_2 \alpha_2+...$ 多个粒子颜色加权求和
```python

```

### 1.4.3 模型优化
- 一个问题：采集粒子时，前面时均匀采样64个点，均匀采样可能采样到无效区域（空白区域和遮挡区域），我们希望有效区域多采样，无效区域少采样。![](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/paper/3DGS/base/3dgs/8.png)
- 解决方法：根据概率密度进行二次采样
    1. 将NeRF分成两个模型：
        - 粗模型：输入均匀采样粒子，输出密度
        - 细模型：根据密度，二次采样
    2. 粗模型和细模型结构相同，最后采用的是二次采样的粒子作为输入
        1) 首先根据公式$\hat C = \sum_ { n = 0 } ^ N C_ n \alpha_ n ( 1 - \alpha_ 0 ) ( 1 - \alpha_ 1 )...( 1 - \alpha_ { n - 1 } )$ 取粒子颜色前的权重做softmax：$w = \alpha_ n ( 1 - \alpha_ 0 ) ( 1 - \alpha_ 1 )...( 1 - \alpha_ { n - 1 } ) , \hat w_i = \frac{ w_ i }{ \sum _ {j = 1 } ^ {N _ c } w_ i}$
        2) 此时，新的权重和为1，可看作概率密度函数，假设分成三个区间，权重分别为：$w=[0.1, 0.7, 0.2]$
        3) 取概率密度函数的CDF（概率累积到当前位置一共是多少）：$\mathrm{CDF}=[0, 0.1, 0.8, 1.0]$ ，然后随机生成一个数，例如0.6，看它落在哪个区间说明应该在哪个区间采样。
        4) 具体采样操作：取CDF的反函数invert，用均匀分布drand48()生成一个随机数：invert(drand48)=r，得到的r就是符合pdf分布的随机数。
    3. 对于每条光线，重新采样128个粒子，与之前的64个粒子加在一起，即每条光线采样192个粒子。
## 1.5 训练流程
### 1）前处理
1. 将图片中的每个像素通过相机模型找到对应的射线
2. 每条射线上进行采样，得到64个粒子
3. 对1024\*64个粒子进行位置编码
     -  位置坐标$x,y,z$ -> 63D
     - 方向向量$x', y', z'$ -> 27D
### 2）粗模型处理
1. 使用8层MLP
2. 输入位置[1024,64,63]、方向[1024,64,27]
3. 输出[1024,64,4]（$rgb \sigma$）
### 3) 后处理
1. 根据粗模型的输出，对射线进行二次采样
2. 每条射线最终192个粒子
### 4）细模型处理
1. 模型网络不变，还是8层MLP
2. 输入为[1024, 192, 63] 和 [1024, 192, 27]
3. 输出为[1024, 192, 4]
### 5）体渲染
1. 输出射线像素颜色
### 6）损失函数
- $L=\sum_ {r \in R} || \hat C ( r ) - C ( r ) ||^ 2 _ 2$
    - R是采样射线条数（1024）
    - GT是射线r对应的那个像素的RGB
    - 预测颜色与GT颜色的MSE损失
## 1.6 推理
1. 输入：h*w条射线上分别采样64个点
2. 输出：h\*w\*192\*4
3. 根据4这个维度（$RGB\sigma$）进行体渲染


---
# 2. 3DGS
## 2.1 3DGS是什么
- 一种3D表征，类似mesh，点云，NeRF； 他使用3D高斯基元来**显式**存储3D信息
    - 在点云的基础上给每个点云扩展成高斯球，然后进行循环优化。
- 3DGS与NeRF一样不具有泛化能力，需要针对每个场景进行训练，来一次训一次，不是feed forward模型。
## 2.2 3DGS模型结构
- 3DGS的原理就是将点云扩展成一个个雪球，然后雪球打到成像平面上，融合得到像素颜色。
### 2.2.1 Splatting
#### 1）概念理解

- Splatting是一种体渲染方法：从3D物体渲染到2D平面
    - NeRF中用的体渲染方法叫Ray-casting，它是被动的。我有一张图片，然后我从每个像素出发去找到影响这个像素的发光粒子，最后确定这个像素的颜色。这个过程中主角是像素。
    - splatting是主动的，计算出每个发光粒子如何影响像素点，主角是粒子。
#### 2）Splatting的流程

1. 选选择基元（雪球）表达方式
2. 渲染（抛雪球），从3D投影到2D
3. 合成颜色
### 2.2.2 3D 高斯
#### 1）定义
$$G( x ) = \frac{ 1 }{ \sqrt{ ( 2 \pi )^ k | \Sigma | } } e^ { - \frac{ 1 }{ 2 } ( x - \mu )^ T \Sigma ^ { - 1 } ( x - \mu ) }$$

- $\Sigma$表示协方差矩阵，半正定；$| \Sigma |$ 是其行列式。$k$时维数
- 为什么选择3D高斯椭球作为基元？因为他又很好的数学性质：
    - 仿射变换后高斯核仍然闭合
    - 3D降维到2D后（沿着某一个轴积分）仍然为高斯
#### 2）3D高斯椭球推导
1. 椭球面表达式：$$\begin{aligned} &\frac{ x^ 2 }{ a^ 2 }+ \frac{ y^ 2 }{ b^ 2 } + \frac{ z^ 2 }{ c^ 2 } = 1 \\\\
  &Ax^ 2 + By^ 2+ Cz^2 + 2Dxy + 2Exz + 2Fyz = 1 \end{aligned}$$
  2. 3D高斯表达式：$$  G( x ) = \frac{ 1 }{ \sqrt{ ( 2 \pi )^ k | \Sigma | } } e^ { - \frac{ 1 }{ 2 } ( x - \mu )^ T \Sigma ^ { - 1 } ( x - \mu ) }
  $$
      - 高斯明显不是上面的椭球那种形式啊？它的值是一个概率，他可能是0.1，0.2...，它为什么是一个椭球呢？
  3. 协方差矩阵$$\Sigma = \begin{bmatrix} \sigma_ { x } ^ 2 \space \sigma_{ xy } \space \sigma_ { xz } \\\\ \sigma_ { yx } \space \sigma_{ y } ^ 2 \space \sigma_ { yz } \\\\ \sigma_ { zx } \space \sigma_{ zy } \space \sigma_ { z } ^ 2 \\\\ \end{bmatrix}$$
      - 一维高斯分布的形状由均值和方差决定。二维时，方差就变成了协方差矩阵
      - 该矩阵为**对称矩形**，决定高斯分布形状。对角线上元素为x轴/y轴/z轴的方差，反斜对角线上的值为协方差，表示x和y，x和z...的线性相关性
  4. 从3D高斯表达式到椭圆
      1) $\frac{ 1 }{ \sqrt{ ( 2 \pi )^ k | \Sigma | } }$ 这部分是常数，带变量的部分在指数上面。
      2) 当$(x - \mu )^ T \Sigma^ { -1 } (x  - \mu )$ 中，x取一个值时，整个高斯的概率就求出来了。
      3) 一维时，$\frac{ ( x - \mu )^ 2 }{ \sigma^ 2 } = c$ 
      4) 二维时 ，椭圆方程 $$
      \begin{align}
      (x - \mu )^ T \Sigma^ { -1 } (x  - \mu ) &= c \\\\ 
      ([x,y] - [\mu_ 1, \mu_ 2]) ^T \begin{bmatrix} 
       \sigma_ { x } ^ 2 \space \sigma_{ xy } \\\\ \sigma_ { yx } \space \sigma_{ y } ^ 2 \\\\ 
      \end{bmatrix}^ { - 1 } ([x,y] - [\mu_ 1, \mu_ 2]) &= c \\\\
      \frac{ ( x - \mu_ 1 )^ 2 }{ \sigma_ 1 ^ 2 } + \frac{ ( y - \mu_ 2 )^ 2 }{ \sigma_ 2 ^ 2 } - \frac{ 2 \sigma_{ xy } ( x - \mu_ 1 )( y - \mu_ 2 ) }{ \sigma_ 1 \sigma_ 2 } &= c
      \end{align}
      $$
      5) 三维时，椭球面方程      $$
      \begin{align}
      (x - \mu )^ T \Sigma^ { -1 } (x  - \mu ) &= c \\\\ 
      \frac{ ( x - \mu_ 1 )^ 2 }{ \sigma_ 1 ^ 2 } + \frac{ ( y - \mu_ 2 )^ 2 }{ \sigma_ 2 ^ 2 } + \frac{ ( z - \mu_ 3 )^ 2 }{ \sigma_ 3 ^ 2 } - \frac{ 2 \sigma_{ xy } ( x - \mu_ 1 )( y - \mu_ 2 ) }{ \sigma_ 1 \sigma_ 2 } - \frac{ 2 \sigma_{ xz } ( x - \mu_ 1 )( z - \mu_ 3 ) }{ \sigma_ 1 \sigma_ 3 } - \frac{ 2 \sigma_{ yz } ( y - \mu_ 2 )( z - \mu_ 3 ) }{ \sigma_ 2 \sigma_ 3 } &= c
      \end{align}
      $$
      6) 每个$G(x , \mu, \Sigma) \in [\times , \times]$ 对应一个c，对应一个椭球面方程。因此3D高斯定义了一个实心的椭球，大椭球壳套小椭球壳，每个椭球壳对应一个概率。
#### 3）协方差控制椭球形状
1. 标准高斯分布$\mathbf{x} \sim N ( \vec{ 0 } , I)$
    - 均值[0,0,0]
    - 协方差矩阵$\begin{bmatrix} 1 &0 &0 \\\\ 0 &1 &0 \\\\ 0 &0 &1 \\\\ \end{bmatrix}$
2. 对标准高斯分布进行仿射变换  $$
  \begin{align}
  \mathbf{ w } &= A \mathbf{ x } + b \\\\
  \mathbf{ w } &\sim N ( A \mu + b , A \Sigma A^ { T })
  \end{align}
  $$
  3. 均值变为$A \mathbf{ x } + b$，改变了位置，协方差改变了形状
  4. 任意高斯可以看作是标准高斯通过仿射变换得到，也就是球经过仿射变换后变成椭球
#### 4）协方差通过旋转与缩放表达
1. 从标准高斯放射变换得到目标高斯：  $$
  \begin{align}
  \mathbf{ w } &= A \mathbf{ x } + b \\\\
  \mathbf{ w } &\sim N ( A \mu + b , A \Sigma A^ { T })
  \end{align}
  $$
  2. 放射变换的定义就是线性变换 + 平移组合，平移部分是b，线性变换部分是A。线性变换可以分解成：旋转、缩放、剪切，因此$A = RSV$ ，带入上式协方差部分$$  \begin{align}
 A \Sigma A^ { T } &=(RSV)\Sigma(V^TSR^T) \\\\
 &=(RSV)I(V^TSR^T) \\\\
  &=(RSV)(V^TSR^T) \\\\
   &=(RS)(SR^T) \\\\
   &=(RS)(S^TR^T) （S是对角矩阵） \\\\
  \end{align}
  $$
  3. 上式表明了协方差可以表达成旋转+缩放。那么如果已经知道协方差矩阵了，怎么求$R、S$呢？
      1) 通过特征值分解将协方差分解成$Q \Lambda Q^ T$ 
      2) $\Lambda$是一个对角矩阵，他的对角线元素是他的特征值：$\begin{bmatrix} S_0 &0 &0 \\\\ 0 &S_1 &0 \\\\ 0 &0 &S_ 2 \end{bmatrix}$
      3) 将中间的$\Lambda$开方：$\Sigma = Q \Lambda^{ \frac{ 1 }{ 2 }} \Lambda^{ \frac{ 1 }{ 2 }} Q^ T$ 
## 2.3 训练流程
### 2.3.1 点云初始化
#### 1）点云数据
- 3DGS通常采用点云数据进行初始化，数据集通常包含如下结构：
```
<location>
|---images        //存放图片，需要编号
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse       //存放colmap点云文件
    |---0
         |---cameras.bin     
         |---images.bin       
         |---points3D.bin    
    |---1
         |---...
```

##### cameras.bin
- 相机内参，不一定只有一条，后面的images.bin里每个图片都有一个CAMERA_ID字段，指向这里的一条记录
```shell
# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# Number of cameras: 1
#ID model    w     h      fx         fy         cx     cy
 1  PINHOLE  1959  1090   1159.5881  1164.6601  979.5  545
```
- 相机ID
- 相机模型：3dgs采用的是 理想的小孔成像模型 **PINHOLE**  （透视投影、**无畸变**）
- 图像宽度
- 图像高度
- x焦距：fx=f/sx，`sx` 指的是**像素尺寸（pixel pitch）**：感光器上**每个像素在 x 方向的物理长度**。  单位通常是 **mm/px** 或 **µm/px**（毫米/像素、微米/像素）。 有了物理焦距 `f`（mm），把长度换成“像素”要除以像素尺寸。  早期或特殊传感器可能**像素非正方形**（$x_ x \neq s_ y$）；或者图像后期做了**非等比缩放**。这都会导致 。$f_ x \neq f_ y$  ，现代相机多为**方形像素** , 差异常来自标定噪声或裁剪/缩放。 
- y焦距
- x主点
- y主点
##### images.bin
- 相机外参，表示如何将每张图片转换到相机坐标系，每张图片对应一份，一份对应两行
```shell
# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)  //每张图上有多个特征点，每个特征点用三元组表示，x,y表示像素位置（原点在左上），point3d_id为对应的三维点编号（见points3D.txt），-1 表示“这个 2D 点没有成功三角化成 3D 点”（只在匹配里出现，但没形成 3D 点）。
# Number of images: 301, mean observations per image: 4223.0996677740868 
#id  QW    QX     QY     QZ    TX    TY    TZ    c_id  img_name
 301 0.996 -0.035 -0.080 0.003 2.520 0.445 4.564 1     00301.jpg
#x1     y1    3Did
 1040.9 246.4 29396 982.9 258.3 35075 994.1 261.3 37804 985.1 261.9 2
```
- 第一行
    - 旋转四元素
    - 平移向量
    - 相机id
    - 图片名
- 第二行
    - 图片中特征点：每张图上有多个特征点，每个特征点用三元组表示，x,y表示像素位置（原点在左上），point3d_id为对应的三维点编号（见points3D.txt），-1 表示“这个 2D 点没有成功三角化成 3D 点”（只在匹配里出现，但没形成 3D 点）。
##### points3D.bin
- 3D点云参数
```shell
# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)    
# Number of points: 182686, mean track length: 6.9581303438687145
#3Did   x    y   z    r  g  b  err img 2Did img 2Did
 206697 1.37 0.5 1.58 20 20 25 1.8 115 8481 117 6815 118 6817 109 10958 110 8002 111 8045 225 8679 112 8807 113 8767 114 9536 115 9310 116 8313 117 7419 119 6948 120 6128 121 6617 122 6907 123 6780 229 8414 226 10631 228 11195 227 7190 124 6885 125 4658 126 4875
```
- 3D点id
- 3D点位置
- 3D点颜色
- 重投影误差：重投影误差的平均/均方根（像素单位）
- 轨迹：由一系列二元组（image_id, point2d_idx）组成，一个二元组表示这个 3D 点被哪张图片的第几个 2D 特征看到（“轨迹”的一次观测），image_id来自images.txt，point2d_idx是特征点id（在一张图像上通过特征检测器（默认 SIFT）找到的兴趣点/关键点），来自images.txt的第二行

#### 2）3D高斯基元
##### 属性
- 每个3D高斯基元有如下属性：
    - 均值：表示位置
    - 协方差：表示形状
    - 颜色
    - 不透明度
- 初始点云提供了位置与颜色，然后根据KNN算法计算当前点与邻居点的距离，以此作为半径初始化正球形高斯
##### 球谐函数
- 3DGS中的颜色并不只是用RGB表示。由于3DGS跟NeRF一样也是采用粒子假设和体渲染，同一个粒子从不同角度反射的光是不一样的，因此3DGS用球谐函数来表达颜色。这样就把颜色和观测角度建立了联系，颜色就不会单一了。$$
\begin{align}
f( t ) \approx &\sum_ l \sum_ { m = -l } ^ l c_ l ^ m y_ l ^ m ( \theta, \phi) \\\\
= &c_0^0 y_ 0 ^ 0 +\\\\
&c_ 1 ^ { -1 } y_ 1 ^ { -1 } +c_ 1 ^ 0 y_ 1 ^ 0 + c_ 1 ^ 1 y_ 1 ^1 + \\\\
& c_ 2^{ -2 } y_ 2^{ -2 } + c_ 2^{ -1 } + y_ 2^{ -1 } + c_ 2^{ 0 } y_ 2^{ 0 } + c_ 2^{ 1 } y_ 2^{ 1 } + c_ 2^{ 2 } y_ 2^{ 2 } + \\\\
& ...
\end{align}
$$
    - y是与视角相关的参数，可根据3D高斯坐标得到
    - c是3维向量（RGB），通过学习获得
    - 3dgs一共使用3阶级球谐系数，一共1+3+5+7=16个c
    - 3D高斯初始化时使用0阶球谐系数，初始颜色采用点云颜色
    - 训练过程中会逐步提高球谐系数
### 2.3.2 前向渲染 
- 世界坐标系 -> 相机坐标系 -> 归一化坐标系 -> 2D像素坐标系 -> 渲染成图
#### 1）坐标变换与高斯投影 
- 假设一个3D高斯基元：
      - 高斯核中心（均值）：$t_k = [t_ 0,t_ 1, t_ 2]^ T$ 
      - 协方差矩阵：$V_k ''$
      - 高斯核（取高斯分布的指数部分）：$r_ k '' ( t ) = G_ { v_ k '' } ( t - t_k )$
##### 1. 视图变换
- 仿射变换，详见[[3DGS笔记#0.2 视图变换]]，根据**相机外参**的平移向量+旋转四元数 得到$W_{w2c}$ ，变换后：
    - 高斯核：$r_ k ' (u ) = G_ { v_ k '} ( u - u_ k )$
    - **均值（线性变换）：$u_ k = W_{w2c} \cdot t_k + d = [u_ 0, u_ 1, u_ 2]^ T$**
    - **协方差矩阵（类似方差平方变换）：$V_k' = W_{w2c} \cdot V_k'' \cdot W_{w2c}^ T$** 
##### 2. 投影变换
- 非仿射变换，详见[[3DGS笔记#0.3.2 透视投影]]，将视锥体压成立方体，然后归一化压成立方体，然后透视除法：$m(t) := \frac{1}{z} M_ { ortho } M _ { persp \rightarrow ortho}$，这个过程是非线性的，且有形变
    - 高斯核：$r_ k ( x ) = G_ { V_ k } ( x - x_ k )$
    - **均值（对于一个点可以直接应用）：$x_ k = m( u_ k ) = \frac{1}{z} M_ { ortho } M _ { persp \rightarrow ortho} \cdot u_k= [x_ 0, x_ 1, x_ 2]^ T$** 
    - **协方差矩阵（非仿射变换，不能直接应用过来）**：？
###### 基于雅可比矩阵的投影变换
- 由于透视投影是非仿射变换，因此不能直接将转换矩阵用到协方差上。在此需要引入雅可比矩阵。
1.  雅可比矩阵做的是泰勒展开，线性逼近
2. 举例：假设在坐标系中有一个点（x，y），对他进行一个非仿射变换：    $$
    \begin{align} 
    f_ 1 ( x ) = x + sin( y ) \\\\
    f_ 2( y ) = y + sin ( y )
    \end{align}
    $$    ![](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/paper/3DGS/base/3dgs/19.png)
    1) 上述变换将平面直角坐标系映射成了一个弯曲的坐标系，(关注某一个点，比如（-2，1）这个点变到了黄色框这个位置，展开这个小框，中心点是（-2，1），其他附近的变换是非线性变化）
    2) 采用微积分的思想，假设把黄色框框进一步缩的很小小，它附近的变换可以近似成一个线性的变换：   ![](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/paper/3DGS/base/3dgs/20.png)
    3) 也就是说确定了一个点$x_ 0 , y_ 0$，在他附近就可以通过导数去进行线性的逼近了。反映到一维坐标系，假设$f(x)=x^2$ , 我想要知道$f(2.1)$ 是多少，真实值是4.41。如何用雅可比矩阵（导数）来逼近呢？
        1. 将f(x)在x0处进行一阶泰勒展开：$f(x) \approx f(x_0)+f'(x_0)(x-x_0)$
        2. 因为刚刚假设的在很小范围内做逼近，因此这里x0取一个离2.1很近的值，比如2
        3. 带入x0=2, x=2.1后得到估计值4.4，与真实值果然很接近。
        4. 上述过程中的一阶倒数就是雅可比矩阵
        5. 对f(x)的均值做估计：$$\begin{aligned}E[f(X)]&≈E[f(x_0​)+f′(x_0​)(X−x_0​)]\\\\&=f(x_0​)+f′(x_0​)E[X−x_0​]\\\\&=f(x_0​)+f′(x_0​)(E[X]−x_0​)\end{aligned}​$$
        6. 对f(x)的方差做估计：$$\begin{aligned}Var(Y)&=E[(Y−E[Y])^2]\\\\&\approx E[(f(x_0​)+f′(x_0​)(X−x_0​) - f(x_0​)-f′(x_0​)(E[X]−x_0​))^2] \\\\&=E[(f′(x_0​)(X−E[X]))^2] \\\\&=[f′(x_0​)]^2 E[(X−E[X])^2]\\\\&=[f′(x_0​)]^2Var(X)​ \end{aligned}$$
        7. 也就是说对于方差使用非仿射变换相当于对他使用这个非仿射变换的雅可比矩阵。
    4) 回到例子，他对应的雅可比矩阵就是$$
    J = \begin{bmatrix} \frac{ df_ 1 }{ dx } &\frac{ df_ 1 }{ dy } \\\\ \frac{ df_ 2 }{ dx } &\frac{ df_2 }{ dy } \end{bmatrix} = \begin{bmatrix} 1 & cos( y ) \\\\ cos( x ) &1 \end{bmatrix}
    $$
3. 对透视投影部分的变换矩阵：$\frac{1}{z} M_ { ortho } M _ { persp \rightarrow ortho}$ 求雅可比矩阵$J$ , 对协方差应用透视投影变换就相当于：**协方差矩阵** = $V_ k = J V_k ' J^ T = JW_{w2c}V_k''W_{w2c}^ T J^ T$ , 具体展开时带入协方差矩阵的x0是高斯中心。
4. 实际上，代码在计算雅可比矩阵的时候是一步到位的，直接从相机坐标系，不需要做正交投影压缩到NDC空间，直接用$\frac{1}{z} M _ { persp \rightarrow ortho}$ 以及视口变换求雅可比。这么做的原因可能是均值需要绝对位置，协方差只描述中心附近的相对偏移。
    1) 对相机坐标的一个点（x,y,z）使用透视转正交变换：  $$
  \left (  \begin{matrix} n&0 &0 &0 \\\\
  0 &n &0 &0 \\\\ 0 &0 &n+f &-nf \\\\
  0 &0 &-1 &0
  \end{matrix} \right) \times \left (  \begin{matrix} x \\\\
  y \\\\ z \\\\
  1
  \end{matrix} \right) = \left (  \begin{matrix} nx \\\\
  ny \\\\ ( n+ f) z -nf \\\\
  -z
  \end{matrix} \right) =\left (  \begin{matrix} \frac{ nx }{ -z } \\\\
  \frac{ ny }{ -z } \\\\ (n+f)-\frac{ nf }{ -z } \\\\
  1
  \end{matrix} \right)
  $$
    2) 我们发现变换关系是：  $$
  \begin{bmatrix} f_ 1( x ) \\\\ f_ 2( y ) \\\\ f_ 3( z ) \end{bmatrix} = \begin{bmatrix}  \frac{ nx }{ -z } \\\\ \frac{ ny }{ -z } \\\\ ( n+f ) - \frac{ nf }{ -z } \end{bmatrix}
  $$
    3) 因此可以求雅可比矩阵（通常带入的是3d高斯的中心点）：$$
  J = \begin{bmatrix} \frac{ n }{ -z } &0 &-\frac{ nx }{ z^ 2 } \\\\ 0 &\frac{ n }{ -z } &-\frac{ ny }{ z^2 } \\\\ 0 &0 &-\frac{ nf }{ z^ 2 }   \end{bmatrix}
  $$
    4) 再应用视口变换，转换到像素空间：$$
  J = \begin{bmatrix} \frac{ f_x }{ -z } &0 &-\frac{ x f_x }{ z^ 2 } \\\\ 0 &\frac{ f_y }{ -z } &-\frac{ yf_y }{ z^2 } \\\\ 0 &0 &0   \end{bmatrix}
  $$
##### 3. 视口变换
- 详见[[3DGS笔记#0.4 光栅化]]，视口变换只需要对高斯核的中心去做，不用对协方差做
#### 2）高斯光栅化与 Alpha 合成 
- 光栅化部分类似NeRF，采用$\alpha - belding$进行颜色合成
##### 1. 分块
1. 选择目标视角
2. 多线程并行执行，将该视角图片划分为多个$16\times16$ 的像素块（tile）
3. 将3D高斯进行坐标变换后到了2D图片范围（z轴忽略），根据投影的2D高斯分布的$3\sigma$ 椭圆长轴半径作为半径画圆，以这个圆的外接矩形作为该高斯可以覆盖的区域（足迹）。
4. 标记这个高斯可以投影到哪个tile，如果一个高斯的足迹覆盖了多个tile就把他复制多份，以tile为单位进行分桶
![](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/paper/3DGS/3dgs_paper/3dgs_tiles.png)
##### 2. 排序
1. 按照块号，深度（z轴），对所有分桶复制后的高斯排序![](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/paper/3DGS/3dgs_paper/3dgs_sort.png)
##### 3. 光栅化
1. 根据排序后的深度先后顺序进行渲染：$$C=\sum_{i=1}^N(c_i\cdot\alpha_i\cdot T_{i-1})$$
      - $c_ i$：通过球谐函数算的RGB值（可学习）
      - $\alpha_i$：第i个高斯的不透明度（范围[0,1]，值越大，对像素的遮挡越强）（可学习）。在一个tile中，高斯中心的不透明度是高斯本身的，在其他地方要乘以一个系数（这个系数根据2D高斯分布计算出来，距离均值的地方越近就越接近1）
          - $\alpha_i(p)=o_i\cdot exp(-\frac{1}{2}(p-\mu_i)^T\Sigma_i^{-1}(p-\mu_i))$ 
      - $T_{i-1}$：光线穿过前i-1个高斯后剩余的比例（可根据$\alpha_i$ 累积计算）
### 2.3.3 渲染损失
$$L=(1-\lambda)L_1 + \lambda L_{D-SSIM}$$
- $L_1=\frac{1}{3HW}\sum_{c=1}^3\sum_{x=1}^W\sum_{y=1}^H|I_c(x,y)-I_c'(x,y)|$ 
- $L_{D-SSIM}=1-SSIM$ 
- $\lambda$ 默认为0.2
### 2.3.4 反向传播与参数优化 
- 通过可微Gaussian rasterizer，把梯度反向传播到每个 Gaussian 的参数。前面协方差矩阵的推导就是为了可微分![](https://cdn.jsdelivr.net/gh/lankeren035/lankeren035.github.io@source/themes/yilia/source/img/paper/3DGS/3dgs_paper/3dgs_back.png)
### 2.3.5 自适应密度控制 
- 每迭代一定的步数（默认100）之后会进行自适应控制
#### 2.3.5.1 高斯克隆 
1. 计算loss对于当前高斯的2D空间梯度$g_i=||\frac{\partial L}{\partial \mu_i^{2D}}||_2$ ，也就是其**投影中心在屏幕空间 x,y 方向的梯度**。并在多个 iteration / view 上累积，最终取平均。
2. 如果梯度较大（说明当前位置与最优位置差距很大，当前 loss 对移动这个 Gaussian 的屏幕空间位置非常敏感，这附近的表示还不充分）
3. 如果当前高斯3D协方差尺度很小$max(s_x,s_y,s_z)\leq p \cdot E$（其中E是场景尺度，p默认0.01），说明延展不足，将其复制。
    - 直观上，一个 Gaussian 已经很小，但这个区域仍然有较大的优化需求。继续把一个 Gaussian 缩小意义不大，更合理的是增加 Gaussian 数量。
4. 完全复制，高斯位置也一样，随着后面的梯度优化过程两个高斯会慢慢区别开。新增 Gaussian 的 Adam optimizer state 是新建的，moment的m和v都是0，而原 Gaussian 已经具有之前训练积累的 Adam moments。
#### 2.3.5.2 高斯分裂 
1. loss对于当前高斯的2D空间梯度较大（说明当前位置与最优位置差距很大）
2. 如果当前高斯3D协方差尺度很大，说明重建过度，将其分割并缩小
3. 新 Gaussian 的中心是从**原 Gaussian 自己的 3D 椭球分布内部随机采出来的**。新生成2个高斯，复制原来的颜色、opacity、rotation 等参数；新 Gaussian scale 缩小（三轴都：$s_{new}=\frac{s_{old}}{0.8\times2}$）；
4. 删除原来的大 Gaussian。
#### 2.3.5.3 高斯剪枝 
- 当不透明度小于某个阈值（0.005）时，剔除
- 屏幕空间过大（$r_i^{2D}>20$），剔除
- 世界空间过大（$max(s_i)>0.1E$，其中E是场景尺度），剔除
#### 2.3.5.4 不透明度重置 
1. 每3000步执行一次opacity reset：$\alpha_i \leftarrow min(\alpha_i, 0.01)$ 

- 为什么要重置 opacity？因为训练过程中一些 Gaussian 可能很早就获得较高 opacity，从而遮挡后方 Gaussian，导致场景结构过早固化。reset之后所有 Gaussian 又需要重新通过后续优化证明自己的贡献：有用 Gaussian → opacity 再升高；没用 Gaussian → opacity 保持很低，最后被 pruning。训练循环确实把 opacity reset 和 densification/pruning 交替执行。

# 3. COLMAP与SfM

- SfM（Structure from Motion，运动恢复结构）：**一套计算机视觉算法理论 / 技术流程，不是软件** ，输入一堆**不同视角、有重叠区域的 2D 照片**，同时求解两件事：
    1. **Motion（运动）**：每一张照片拍摄时相机的内参（焦距、畸变）、外参（相机在 3D 空间的位置 + 朝向）；
     2. **Structure（结构）**：场景里物体的三维稀疏点云。SfM 只输出**稀疏点云**（只有特征点，很稀疏）；想要稠密完整模型，需要后续 MVS（多视图立体匹配）。
 - 标准 SfM 完整流程（三步）
     1. 特征提取（SIFT 等局部特征）
     2. 图像匹配 + 几何校验（筛除错误匹配点）
     3. 增量重建：初始化模型→逐帧注册相机→三角化 3D 点→BA 光束平差全局优化
- **COLMAP 是一套开源、工业级、学术界通用的完整软件工具**，由慕尼黑工大开发，是实现 SfM+MVS 的标准工具链。- 支持 GUI 可视化操作 + 命令行批量处理；内置**增量式 SfM**作为核心稀疏重建模块；同时自带 MVS 稠密重建模块，能从稀疏点云生成稠密点云、三维网格；几乎所有 3D 重建、NeRF/3DGS 高斯泼溅、SLAM 论文都会用它预处理数据集。



# 4. 4DGS

# 5. 3DGS进阶
### 5.1 FastGS
- 通过重新设计裁剪与致密化策略，让训练过程中 Gaussian 数量始终维持在较低水平，从而让每一步训练都更便宜。
- 原始3dgs的训练主要有两件事情：1. 参数优化  2. ADC（Densification+Pruning）。原始3dgs的densification基本看image-space position gradient，梯度大，就认为这个 Gaussian 所在区域没拟合好，于是 clone/split。但是某个视角梯度大”并不意味着这个 Gaussian 在三维意义上真的需要 densify，它可能只是：某一视角遮挡；某一个 view 出现局部误差；高频纹理导致梯度大；已经存在别的 Gaussian 可以解释这个区域。于是 Gaussian 会不断膨胀到几百万。
#### 5.1.1 VCD - Multi-view Consistent Densification
- 原始3dgs仅根据梯度判断是否densification，这里**进一步增加**multi-view error。
1. 一次性取k（10）个视角算RGB L1 error然后颜色通道平均：$e_{u,v}^j=\frac{1}{C}\sum_c|r_{u,v}^{j,c}-g_{u,v}^{j,c}|$ 
    - u,v表示一个像素
    - j表示一个视角
    - C表示通道
    - r表示某一个视角某一个像素某一个通道的渲染图颜色
    - g表示GT图
2. 对上面求得的逐像素error进行min-max normalization，然后根据预定义的阈值得到一个**mask**，mask=0的地方表示像素误差比较小，拟合的不错；mask=1的地方表示重建的比较差的像素。
3. 对每个高斯，将他们投影到上面的k个视角下。统计他的足迹覆盖了多少误差较大的像素：$$s_d^i = \frac{1}{K}\sum_{j=1}^K\sum_{p\in\Omega_i^j} 1(M_{mask}^j(p)=1)$$
    - K表示训练一次性取的视角
    - j表示某个视角
    - $\Omega_i^j$ 表示高斯i在视角j下的足迹
    - p表示足迹里面包含的像素
4. 如果高斯i空间梯度大的同时$s_d^i$ 计算出来大于预定义的阈值，才允许densification。
#### 5.1.2 VCP - Multi-view Consistent Pruning
- 对应的裁剪策略也需要优化。
1. 计算某个视角j的整体error$$E_{photo}^j=(1-\lambda)L_1^j+\lambda(1-L_{ssim}^j)$$
2. 对每个高斯，将他们投影到上面的k个视角下。统计他的足迹覆盖了多少误差较大的像素然后乘以该视角的整体误差：$$s_p^i=\sum_{j=i}^K[\sum_{p\in\Omega_i^j}1(M_{mask}^j(p)=1) \cdot E_{photo}^j]$$
    - 其实就是在VCD基础上乘以像素误差
3. 如果$s_p^i$ 很高，说明一个 Gaussian 在**多个整体重建很差的视角里**，又长期落在 high-error region，那么这个 Gaussian 更可能没有有效帮助 reconstruction，甚至与 reconstruction degradation 相关。所以不是将他增值而是裁剪掉。
4. 为什么 VCD 高分要增加，VCP 高分反而要删除？
    - VCD不是看到 error 大就直接 densify。他还要满足梯度条件，当前 Gaussian 梯度已经说明“它想发生明显变化”，同时多个视角又说明“这里确实一直没拟合好”。
#### 5.1.3 Compact Box
- 原始3dgs以高斯分布的3sigma椭圆的长轴半径为半径的圆的外接矩形作为覆盖区域。
- FastGS先设定不透明度阈值，然后根据这个阈值结合高斯分布的方程，算出一个动态的椭圆长轴半径为k sigma， 这个椭圆范围内高斯不透明度大于阈值。以这个长轴半径作为规范。这样就可以根据不透明度动态选择覆盖范围。减少分桶排序复制的开销以及不透明度过低的区域。
