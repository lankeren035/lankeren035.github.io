---
title: "readColmapSceneInfo"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/scene.dataset_readers.py.readColmapSceneInfo"
hexo-path:
---

## 1. 输入

| 参数名            | 类型        | 解释                                       |
| -------------- | --------- | ---------------------------------------- |
| path           | -         | 数据集路径                                    |
| images         | -         | 数据集路径colmap下面的images文件夹                  |
| depths         | -         | -                                        |
| eval           | bool      | -                                        |
| train_test_exp | -         | -                                        |
| llffhold       | default=8 | LLFF 常见的 hold-out 策略，从所有图像中按固定间隔抽一部分做测试集 |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| - | - |

## 3. 操作逻辑

1. 读取数据集中如下两个文件：
	- 外参文件：sparse/0/images.bin [[3dgs_read_extrinsics_binary]]
	- 内参文件：sparse/0/cameras.bin [[3dgs_read_intrinsics_binary]]
2. 读取深度图：##todo
3. 划分测试集：
	- 默认：取出每个相机对应的拍摄图片名字，将名字排序，每隔llffhold张取一张作为测试集
	- 手动指定：提供sparse/0/test.txt
4. 把读出来的相机外参、内参、图像路径、深度图路径、训练/测试标记，整理成 3DGS 自己统一使用的 `CameraInfo` 列表。[[3dgs_readColmapCameras]]
5. 将CameraInfo列表中的数据划分为训练集和测试集
6. 根据训练相机位姿，获取场景归一化参数：平移和缩放。[[3dgs_getNerfppNorm]]
7. 读取点云文件[[3dgs_fetchPly]]
8. 打包返回场景信息 [[3dgs_SceneInfo]]
## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_dataset_readers.py\|dataset_readers.py]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | - |
