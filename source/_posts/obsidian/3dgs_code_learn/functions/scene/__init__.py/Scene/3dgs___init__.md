---
title: "__init__"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/scene.__init__.py.Scene.__init__"
hexo-path:
---

## 1. 输入

| 参数名               | 类型            | 解释      |
| ----------------- | ------------- | ------- |
| args              | ModelParams   | 数据集模型参数 |
| gaussians         | GaussianModel | 高斯模型类实例 |
| load_iteration    | default=None  | -       |
| shuffle           | default=True  | -       |
| resolution_scales | default=[1.0] | -       |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| - | - |

## 3. 操作逻辑

1. 确定如下属性：
	```
	model_path = args.model_path #输出模型路径
	loaded_iter = None
	gaussians = gaussians #场景中的高斯实例
	```
2. 读取场景信息：[[3dgs_readColmapSceneInfo]]
3. 如果不加载预训练模型，则将初始数据记录到实验目录：
	- 点云文件
	- 相机参数（测试在前训练在后，json保存）
4. 遍历所有图片分辨率（这个分辨率不是数据集的 1，1/2分辨率，而是当前这个类构造函数的resolution_scales）：
	- 对训练集测试集分别打包相机参数[[3dgs_cameraList_from_camInfos]]
	- 最终train_cameras\[resolution_scale]/ test_cameras\[resolution_scale]的相机参数格式： \[ [[source/_posts/obsidian/3dgs_code_learn/modules/scene/cameras.py/3dgs_Camera|Camera1]], [[source/_posts/obsidian/3dgs_code_learn/modules/scene/cameras.py/3dgs_Camera|Camera2]],... ]
5. 从点云数据和训练相机信息中初始化高斯模型[[3dgs_create_from_pcd|create_from_pcd(点云数据，训练相机信息，场景尺度)]]
## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[source/_posts/obsidian/3dgs_code_learn/files/scene/3dgs___init__.py\|__init__.py]] |
| 所属类 | [[3dgs_Scene\|Scene]] |
| 命名空间 | - |
| 类型 | - |
