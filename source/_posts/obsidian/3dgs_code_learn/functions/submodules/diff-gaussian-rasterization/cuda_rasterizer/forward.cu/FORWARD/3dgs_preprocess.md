---
title: "preprocess"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/submodules.diff-gaussian-rasterization.cuda_rasterizer.forward.cu.FORWARD.preprocess"
hexo-path:
---

## 1. 输入


| 参数名            | 类型               | 解释                                                                         |
| -------------- | ---------------- | -------------------------------------------------------------------------- |
| P              | int              | 点数                                                                         |
| D              | int              | 当前启用的球谐阶数                                                                  |
| M              | int              | 球谐系数总个数（3阶为16个）                                                            |
| means3D        | const float*     | （P, 3)，每个点的3D坐标                                                            |
| scales         | const glm::vec3* | 协方差缩放                                                                      |
| scale_modifier | const float      | 对所有高斯的尺度做一个统一调节。值大，高斯更“胖”；值小，高斯更“瘦”，代码中默认=1                                |
| rotations      | const glm::vec4* | 协方差旋转                                                                      |
| opacities      | const float*     | （P, 1)，每个点的不透明度，初始0.1                                                      |
| shs            | const float*     | 球谐系数【p, 16, 3】                                                             |
| clamped        | bool*            | 来自[[3dgs_GeometryState|GeometryState]]，记录 RGB 三个通道是否被 clamp 过             |
| cov3D_precomp  | const float*     | python预计算的3D协方差矩阵 【N, 3, 3】（如有）                                            |
| colors_precomp | const float*     | python预计算的高斯颜色【N, 3】（如有）                                                   |
| viewmatrix     | const float*     | 世界坐标系转换到相机坐标系                                                              |
| projmatrix     | const float*     | 完整投影矩阵，把3D视锥投影到NDC，x 和 y 轴的范围是 \[-1, 1]，z 轴范围通常是 \[0, 1]                   |
| cam_pos        | const glm::vec3* | 相机中心在世界坐标系中的位置                                                             |
| W              | const int        | 图像宽度                                                                       |
| H              | int              | 图像高度                                                                       |
| focal_x        | const float      | x轴焦距                                                                       |
| focal_y        | float            | y轴焦距                                                                       |
| tan_fovx       | const float      | 水平视场角（半角）的切线值                                                              |
| tan_fovy       | float            | 垂直视场角（半角）的切线值                                                              |
| radii          | int*             | 高斯屏幕半径（外接矩形）                                                               |
| means2D        | float2*          | 来自[[3dgs_GeometryState|GeometryState]]，每个高斯投影到图像平面后的 2D 中心位置              |
| depths         | float*           | 来自[[3dgs_GeometryState|GeometryState]]，每个高斯的深度值（后面要按 **tile + depth** 排序） |
| cov3Ds         | float*           | 来自[[3dgs_GeometryState|GeometryState]]，3D 协方差矩阵的压缩表示（对称矩阵保存6个数）           |
| rgb            | float*           | 来自[[3dgs_GeometryState|GeometryState]]，高斯颜色                               |
| conic_opacity  | float4*          | 来自[[3dgs_GeometryState|GeometryState]]，椭圆二次型的 3 个参数与不透明度                  |
| grid           | const dim3       | 将图片划分成多个tiles的实例                                                           |
| tiles_touched  | uint32_t*        | 来自[[3dgs_GeometryState|GeometryState]]，每个高斯一共覆盖了多少个 tile                  |
| prefiltered    | bool             | 是否启用预过滤                                                                    |
| antialiasing   | bool             | 是否启用抗锯齿                                                                    |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| void | - |

## 3. 操作逻辑

1. 在CPU侧调用GPU侧的渲染核函数，启动(P + 255) / 256个线程block，每个block 256个线程：[[source/_posts/obsidian/3dgs_code_learn/functions/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu/3dgs_preprocessCUDA|3dgs_preprocessCUDA]]

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_forward.cu|forward.cu]] |
| 所属类 | - |
| 命名空间 | FORWARD |
| 类型 | namespace_scoped |
