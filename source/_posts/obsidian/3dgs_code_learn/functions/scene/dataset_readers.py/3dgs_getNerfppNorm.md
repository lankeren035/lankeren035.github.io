---
title: "getNerfppNorm"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/scene.dataset_readers.py.getNerfppNorm"
hexo-path:
---

## 1. 输入

| 参数名      | 类型  | 解释       |
| -------- | --- | -------- |
| cam_info | -   | 训练相机信息列表 |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| - | - |

## 3. 操作逻辑

1. 遍历相机信息列表，对于每张图：
	1) 计算W2C矩阵（world to camera）：[[3dgs_getWorld2View2]]
	2) 对W2C求逆得到C2W
	3) 估计场景中心和场景尺度
		- 从C2W列表得到所有图片的的相机中心
		- 对这些相机中心按xyz取平均得到一个整体场景中心点center
		- 计算里整体场景中心点与最远相机中心的距离 \* 1.1 得到场景的大小radius
	4) 返回 -center 与 radius  (用于后续对整个场景进行平移缩放归一化)

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_dataset_readers.py\|dataset_readers.py]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | - |
