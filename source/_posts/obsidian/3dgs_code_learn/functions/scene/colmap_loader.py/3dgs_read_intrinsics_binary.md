---
title: "read_intrinsics_binary"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/scene.colmap_loader.py.read_intrinsics_binary"
hexo-path:
---
- 数据形式

| id   | model | width | height | params               |
| ---- | ----- | ----- | ------ | -------------------- |
| 相机id | 相机模型  | 图片宽度  | 图片高度   | 相机参数：焦距fx/fy，中心cx/cy |

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| path_to_model_file | - | - |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| - | - |

## 3. 操作逻辑

1. 读取相机数量：n
2. 循环遍历n：
	1) 读取相机参数（24字节，iiQQ）：相机id，相机模型id，图像宽度，图像高度
	2) 根据相机模型id，获取该相机属性个数m
	3) 根据属性个数读取对应的字节数（8 $\times$ m字节i）
3. 最终数据形式：

| id   | model | width | height | params               |
| ---- | ----- | ----- | ------ | -------------------- |
| 相机id | 相机模型  | 图片宽度  | 图片高度   | 相机参数：焦距fx/fy，中心cx/cy |

## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_colmap_loader.py\|colmap_loader.py]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | - |
