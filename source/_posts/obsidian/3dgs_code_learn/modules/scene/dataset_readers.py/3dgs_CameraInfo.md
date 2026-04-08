---
title: "CameraInfo"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/scene.dataset_readers.py.CameraInfo"
hexo-path:
---
- 结构体，仅包含属性  [[##2. 属性|属性列表]]

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| - | - | - |

## 2. 属性

| 属性名          | 类型       | 来源函数    | 解释             |
| ------------ | -------- | ------- | -------------- |
| FovX         | np.array | <class> | x轴视场角（0, 3.14） |
| FovY         | np.array | <class> | y轴视场角（0, 3.14） |
| R            | np.array | <class> | 旋转矩阵,【3，3】     |
| T            | np.array | <class> | 平移向量【3】        |
| depth_params | dict     | <class> | 深度参数           |
| depth_path   | str      | <class> | 深度图路径          |
| height       | int      | <class> | 图高             |
| image_name   | str      | <class> | 图像文件名          |
| image_path   | str      | <class> | 图像路径           |
| is_test      | bool     | <class> | 该图是不是测试集图片     |
| uid          | int      | <class> | 相机id           |
| width        | int      | <class> | 图宽             |

## 3. 方法

| 方法名 | 解释 |
|---|---|
| - | - |
