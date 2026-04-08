---
title: "SceneInfo"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/scene.dataset_readers.py.SceneInfo"
hexo-path:
---
- 打包场景信息：

| 属性名                | 类型              | 解释      |
| ------------------ | --------------- | ------- |
| is_nerf_synthetic  | bool = false    |         |
| nerf_normalization | dict            | 场景归一化参数 |
| ply_path           | str             | 点云文件路径  |
| point_cloud        | BasicPointCloud | 点云数据    |
| test_cameras       | list            | 测试相机信息  |
| train_cameras      | list            | 训练相机信息  |

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| - | - | - |

## 2. 属性

| 属性名                | 类型              | 来源函数    | 解释      |
| ------------------ | --------------- | ------- | ------- |
| is_nerf_synthetic  | bool            | <class> |         |
| nerf_normalization | dict            | <class> | 场景归一化参数 |
| ply_path           | str             | <class> | 点云文件路径  |
| point_cloud        | BasicPointCloud | <class> | 点云数据    |
| test_cameras       | list            | <class> | 测试相机信息  |
| train_cameras      | list            | <class> | 训练相机信息  |

## 3. 方法

| 方法名 | 解释 |
|---|---|
| - | - |
