---
title: "Image"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/scene.colmap_loader.py.Image"
hexo-path:
---
- 继承自BaseImage，其数据结构：

| id   | qvec【4】 | tvec【3】 | camera_id | name  | xys【p,2】       | point3D_ids【p,1】 |
| ---- | ------- | ------- | --------- | ----- | -------------- | ---------------- |
| 图片id | 旋转四元数   | 平移向量    | 相机id      | 图片文件名 | 当前图中所有特征点的像素坐标 | 当前图中所有特征点id      |
- 在其基础上增加了一个函数[[source/_posts/obsidian/3dgs_code_learn/functions/scene/colmap_loader.py/3dgs_qvec2rotmat|qvec2rotmat]] 将旋转四元数转换成旋转矩阵

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| - | - | - |

## 2. 属性

| 属性名 | 类型 | 来源函数 | 解释 |
|---|---|---|---|
| - | - | - | - |

## 3. 方法

| 方法名 | 解释 |
|---|---|
| [[source/_posts/obsidian/3dgs_code_learn/functions/scene/colmap_loader.py/Image/3dgs_qvec2rotmat\|qvec2rotmat]] | - |
