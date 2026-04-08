---
title: "loadCam"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/utils.camera_utils.py.loadCam"
hexo-path:
---

## 1. 输入

| 参数名               | 类型   | 解释                               |
| ----------------- | ---- | -------------------------------- |
| args              | -    | 模型参数[[3dgs_ModelParams]]         |
| id                | -    | 序号                               |
| cam_info          | -    | 一条相机信息                           |
| resolution_scale  | -    | 分辨率尺度（来自Scene类初始化[[3dgs_Scene]]） |
| is_nerf_synthetic | bool |                                  |
| is_test_dataset   | bool |                                  |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| - | - |

## 3. 操作逻辑

1. 根据一条相机信息，打开图片文件
2. 根据一条相机信息，打开深度文件：##todo
3. 根据启动命令是否提供了resolution参数：
	- 是：则将图片缩小（resolution\*resolution_scale 倍） （注意，这里原始数据集可能有各种分辨率的图片：image,  image_2，代码读取的永远是image文件夹，读取后根据提供的参数进行缩放）
	- 否（-1）：缩放保证宽度<= 1600
4. 获得最终缩放比例与最终图像分辨率
5. 相机信息最终打包返回[[source/_posts/obsidian/3dgs_code_learn/modules/scene/cameras.py/3dgs_Camera|Camera]]类：[[source/_posts/obsidian/3dgs_code_learn/functions/scene/cameras.py/Camera/3dgs___init__|3dgs___init__]]

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_camera_utils.py|camera_utils.py]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | - |
