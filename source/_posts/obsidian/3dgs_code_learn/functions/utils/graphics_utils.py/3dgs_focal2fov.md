---
title: "focal2fov"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/utils.graphics_utils.py.focal2fov"
hexo-path:
---
- 根据焦距和图片宽高得到视场角：
	$$
	FoV=2arctan(\frac{pixels}{2f})
    $$

## 1. 输入

| 参数名    | 类型  | 解释    |
| ------ | --- | ----- |
| focal  | -   | x/y焦距 |
| pixels | -   | 图片w/h |

## 2. 输出

| 返回类型 | 解释 |
|---|---|
| - | - |

## 3. 操作逻辑

1. 使用公式计算出视场角：
	$$
	FoV=2arctan(\frac{pixels}{2f})
    $$
![[themes/yilia/source/img/obsidian/3dgs/1.png]]
## 4. 信息

| 字段 | 内容 |
|---|---|
| 所属文件 | [[3dgs_graphics_utils.py\|graphics_utils.py]] |
| 所属类 | - |
| 命名空间 | - |
| 类型 | - |
