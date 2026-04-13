---
title: "read_extrinsics_binary"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/scene.colmap_loader.py.read_extrinsics_binary"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| path_to_model_file | - | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| - | - |

## 3. 操作逻辑

1. 读取图片数量num_reg_images，8字节，Q（无符号 8 字节整数）[[source/_posts/obsidian/3dgs_code_learn/functions/scene/colmap_loader.py/3dgs_read_next_bytes|read_next_bytes(文件指针，字节数，Q)]]
2. 循环遍历上面读取到的图片数量：
	- 读取对应的数据
	- 如何读取每个图片的xy：
		1. 一步读取该图片的特征点数量n
		2. 读取n $\times$ 24 个字节得到（每个特征点包含：x,y,point3D_id三个字段，各8字节）
		3. 读取每个特征点的xy，即得到当前图片中所有2D特征点的像素坐标【p,2】
3. 最终得到的数据格式：images{0：图片0的数据，1：图片1的数据}，每条数据格式如下[[source/_posts/obsidian/3dgs_code_learn/modules/scene/colmap_loader.py/3dgs_Image|Image]]

| id   | qvec【4】 | tvec【3】 | camera_id | name  | xys【p,2】       | point3D_ids【p,1】 |
| ---- | ------- | ------- | --------- | ----- | -------------- | ---------------- |
| 图片id | 旋转四元数   | 平移向量    | 相机id      | 图片文件名 | 当前图中所有特征点的像素坐标 | 当前图中所有特征点id      |



## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_colmap_loader.py|colmap_loader.py]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | - |
