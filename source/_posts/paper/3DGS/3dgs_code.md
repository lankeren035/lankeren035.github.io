---
title: "3DGS代码理解"
date: 2025-9-17 10:28:00
tags: [3dgs]
categories: [3dgs]
comment: false
toc: true

---

#
<!--more-->

## 1. 数据集形式

```
<location>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
 |---0
     |---cameras.bin      //相机内参，相机型号表，不一定只有一条，后面的images.bin里每个图片都有一个CAMERA_ID字段，指向这里的一条记录
     |---images.bin       //每张图的位姿 + 2D观测。位姿是什么：相机的外参，给在四元数和平移向量里，在colmap约定下，他表示world -> camear的变换。2D观测是什么：一张图片里检测到的特征点坐标及其与3D点的对应关系
     |---points3D.bin     //稀疏三维点 + 轨迹。场景里被多视角三角化出的 3D 点集合。轨迹 (IMAGE_ID, POINT2D_IDX)：这个 3D 点被哪些图片的第几个2D特征观测到（这就能回指到 images.bin 里那张图的第 POINT2D_IDX 个观测）
     |---points3D.ply     //将稀疏三维点导出的点云文件
```

- 使用colmap将这几个文件变成文本：

  ```shell
  conda install -y -c conda-forge colmap libstdcxx-ng libgcc-ng
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64"
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/libgcc_s.so.1"
  colmap model_converter   --input_path ./data/input/tandt_db/train/sparse/0   --output_path ./data/input/tandt_db/train/sparse/0_txt   --output_type TXT  #注意这里的输出路径需要是已存在的文件夹
  ```


### 1.1 cameras.txt内容

```shell
# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# Number of cameras: 1
1  PINHOLE  1959  1090   1159.5881  1164.6601  979.5  545
│   │        │      │        │          │         │     └─ cy（主点（中心点） y）
│   │        │      │        │          │         └────── cx（主点 x）
│   │        │      │        │          └───────── fy（像素单位）（焦距/像素尺寸）
│   │        │      │        └──────────────────── fx（像素单位）
│   │        │      └───────────────────────────── 图高 H（像素）
│   │        └──────────────────────────────────── 图宽 W（像素）
│   └───────────────────────────────────────────── 相机模型：PINHOLE
└──────────────────────────────────────────────── CAMERA_ID=1
```

-  **整个稀疏模型只有这一种内参/分辨率**，所以每张图都会在 `images.txt` 里引用 `CAMERA_ID=1` 
- 3dgs采用的是 理想的小孔成像模型 **PINHOLE**  （透视投影、**无畸变**）。只需要 4 个内参：`fx, fy, cx, cy`。fx=f/sx，fy=f/sy。 `sx` 和 `sy` 指的是**像素尺寸（pixel pitch）**：感光器上**每个像素在 x / y 方向的物理长度**。  单位通常是 **mm/px** 或 **µm/px**（毫米/像素、微米/像素）。 有了物理焦距 `f`（mm），把长度换成“像素”要除以像素尺寸。  早期或特殊传感器可能**像素非正方形**（$x_ x \neq s_ y$）；或者图像后期做了**非等比缩放**。这都会导致 。$f_ x \neq f_ y$  ，现代相机多为**方形像素** , 差异常来自标定噪声或裁剪/缩放。 

### 1.2 images.txt内容

```shell
# Image list with two lines of data per image:     //每张图片占两行
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME   //Q是四元旋转数（world->camera），T是平移向量，camera_id指向cameras.txt里的那一条
#   POINTS2D[] as (X, Y, POINT3D_ID)  //每张图上有多个特征点，每个特征点用三元组表示，x,y表示像素位置（原点在左上），point3d_id为对应的三维点编号（见points3D.txt），-1 表示“这个 2D 点没有成功三角化成 3D 点”（只在匹配里出现，但没形成 3D 点）。
# Number of images: 301, mean observations per image: 4223.0996677740868   //平均每图大约 4K 多个 2D 点
301 0.996 -0.035 -0.080 0.003 2.520 0.445 4.564 1 00301.jpg
1040.9 246.4 29396 982.9 258.3 35075 994.1 261.3 37804 985.1 261.9 2
```

### 1.3 points3D.txt内容

```shell
# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)     //error是重投影误差的平均/均方根（像素单位），track是轨迹，由一系列二元组（image_id, point2d_idx）组成，一个二元组表示这个 3D 点被哪张图片的第几个 2D 特征看到（“轨迹”的一次观测），image_id来自images.txt，point2d_idx是特征点id（在一张图像上通过特征检测器（默认 SIFT）找到的兴趣点/关键点），来自images.txt的第二行
# Number of points: 182686, mean track length: 6.9581303438687145
206697 1.37 0.5 1.58 20 20 25 1.8 115 8481 117 6815 118 6817 109 10958 110 8002 111 8045 225 8679 112 8807 113 8767 114 9536 115 9310 116 8313 117 7419 119 6948 120 6128 121 6617 122 6907 123 6780 229 8414 226 10631 228 11195 227 7190 124 6885 125 4658 126 4875
```



## 2. 前向传播

- 已有参数：
  - 相机内参：焦距，主点，分辨率：$f_x , f_y , c_x , c_y , H , W$
  - 相机外参：旋转四元数，平移向量：$Q, T$
  - 高斯属性：位置，颜色，半径：$x,y,z,R,G,B,\Sigma$

